// llm-injection graph sites: attention temperature, head lesions, residual-stream noise / layer gain,
// FFN dropout, FFN channel lesions and activation steering.
// Part of LLM Injection Runtime Lab (MIT License).
//
// Pseudo-randomness inside the graph comes in two modes (LLM_INJ_NOISE_MODE):
//   sin  (default) u(x) = sin(K * x + phase_layer), plain ggml element-wise ops, runs on every backend.
//        With a large K it behaves like a bounded, high-frequency hash of each activation, but it is a
//        smooth function of x and therefore not independent of the input.
//   hash a CPU custom op computing gauss(splitmix64(seed, layer, channel, bits(x))). The value bits
//        are hashed, so the noise is uncorrelated with x (independent across channels and values).
//        Custom ops run on the CPU backend; with GPU offload the scheduler copies the tensor.
// Head and channel lesions use small CPU custom ops that write fixed masks (no RNG involved).
//
// When a pharmacokinetic schedule is active, every effect is scaled by m(t) at graph-build time and
// the graph is rebuilt for every token (see llm_inj_graph_reusable); ops are always built, even at
// m = 0, so the graph topology does not change over time.

#include "llm-injection.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>

static const float K_FREQ_NOISE   = 4099.0f;
static const float K_FREQ_DROPOUT = 3571.0f;

static const uint64_t SALT_RESID   = 0x7265736964ULL;
static const uint64_t SALT_DROPOUT = 0x66666e64726f70ULL;

static float layer_phase(uint64_t seed, int il, uint64_t salt) {
    return (float) (6.283185307179586 * llm_inj_unit(llm_inj_hash(llm_inj_hash(seed, salt), (uint64_t) (uint32_t) il)));
}

// ---------------------------------------------------------------------------------------------
// custom ops. userdata is a packed 64-bit value (no allocation, valid for the graph's lifetime):
//   bits  0..31  a (seed low bits, or n_layer for masks)
//   bits 32..43  layer index
//   bits 44..47  kind
//   bits 48..63  q, a value in [0, 1] quantized to 1/65535 (dropout probability or pk factor)

enum op_kind { OP_NOISE = 1, OP_DROPOUT = 2, OP_HEADS = 3, OP_FFN_LESION = 4, OP_STEER = 5 };

static void * pack(uint32_t a, int il, int kind, float q) {
    const uint64_t qq = (uint64_t) std::lround(std::min(1.0f, std::max(0.0f, q)) * 65535.0f);
    const uint64_t v  = (uint64_t) a | ((uint64_t) (il & 0xfff) << 32) | ((uint64_t) (kind & 0xf) << 44) | (qq << 48);
    return (void *) (uintptr_t) v;
}

struct unpacked { uint32_t a; int il; int kind; float q; };

static unpacked unpack(void * ud) {
    const uint64_t v = (uint64_t) (uintptr_t) ud;
    return { (uint32_t) (v & 0xffffffffULL), (int) ((v >> 32) & 0xfff), (int) ((v >> 44) & 0xf),
             (float) (v >> 48) / 65535.0f };
}

// element-wise: noise (standard normal) or inverted-dropout keep mask, from the hash of each value
static void op_hash_elementwise(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * ud) {
    static_assert(sizeof(void *) >= 8, "64-bit pointers required for packed userdata");
    const unpacked u = unpack(ud);
    const int64_t ne0 = a->ne[0], ne1 = a->ne[1], ne2 = a->ne[2], ne3 = a->ne[3];
    const int64_t nr  = ne1 * ne2 * ne3;
    const int64_t dr  = (nr + nth - 1) / nth;
    const int64_t r0  = dr * ith;
    const int64_t r1  = std::min(r0 + dr, nr);
    const float   p   = u.q;
    const float   inv = p < 1.0f ? 1.0f / (1.0f - p) : 0.0f;
    const uint64_t salt = u.kind == OP_NOISE ? SALT_RESID : SALT_DROPOUT;
    for (int64_t r = r0; r < r1; ++r) {
        const int64_t i1 = r % ne1, i2 = (r / ne1) % ne2, i3 = r / (ne1 * ne2);
        const char * src = (const char *) a->data + i1 * a->nb[1] + i2 * a->nb[2] + i3 * a->nb[3];
        float * out = (float *) ((char *) dst->data + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            const float x = *(const float *) (src + i0 * a->nb[0]);
            if (u.kind == OP_NOISE) {
                out[i0] = llm_inj_hash_noise(u.a, u.il, salt, i0, x);
            } else {
                const uint64_t key = llm_inj_hash(llm_inj_hash(u.a, salt), (uint64_t) (uint32_t) u.il);
                uint32_t bits;
                std::memcpy(&bits, &x, sizeof(bits));
                const double unit = llm_inj_unit(llm_inj_hash(key, ((uint64_t) i0 << 32) | bits));
                out[i0] = unit < (double) p ? 0.0f : inv;
            }
        }
    }
}

// 1-D masks over heads or channels (input: arange(0, n))
static void op_mask(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) {
        return;
    }
    const unpacked u = unpack(ud);
    const llm_inj_config & c = llm_inj_cfg();
    const int64_t n = a->ne[0];
    float * out = (float *) dst->data;
    for (int64_t i = 0; i < n; ++i) {
        if (u.kind == OP_HEADS) {
            out[i] = llm_inj_head_gain(c, u.q, u.il, (int) u.a, (int) i, (int) n);
        } else {
            out[i] = llm_inj_ffn_lesioned(c, u.q, u.il, i) ? 0.0f : 1.0f;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// steering directions (control-vector GGUF)

struct steer_set {
    std::string                     file;
    long long                       stamp  = 0;
    bool                            exists = false;
    bool                            ok     = false;
    std::map<int, std::vector<float>> dirs;
};

static std::mutex                              g_steer_mutex;
static std::vector<std::unique_ptr<steer_set>> g_steer_sets; // kept alive: pointers are used during compute

// modification stamp of a file; existence is reported separately because the clock epoch is
// implementation-defined (libstdc++'s file clock makes present-day times negative)
static bool file_stamp(const std::string & file, long long & stamp) {
    std::error_code ec;
    const auto t = std::filesystem::last_write_time(std::filesystem::u8path(file), ec);
    if (ec) {
        stamp = 0;
        return false;
    }
    const auto size = std::filesystem::file_size(std::filesystem::u8path(file), ec);
    stamp = (long long) t.time_since_epoch().count() ^ (long long) (ec ? 0 : size);
    return true;
}

static std::unique_ptr<steer_set> load_steer(const std::string & file, long long stamp, bool exists) {
    auto set    = std::make_unique<steer_set>();
    set->file   = file;
    set->stamp  = stamp;
    set->exists = exists;
    ggml_context * meta = nullptr;
    gguf_init_params params = { /*no_alloc =*/ false, /*ctx =*/ &meta };
    gguf_context * g = exists ? gguf_init_from_file(file.c_str(), params) : nullptr;
    if (g == nullptr) {
        return set;
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        int il = -1;
        if (std::sscanf(name, "direction.%d", &il) != 1 || il < 0) {
            continue;
        }
        const ggml_tensor * t = ggml_get_tensor(meta, name);
        if (t == nullptr || t->type != GGML_TYPE_F32 || ggml_nelements(t) <= 0) {
            continue;
        }
        const int64_t n = ggml_nelements(t);
        std::vector<float> v((const float *) t->data, (const float *) t->data + n);
        bool finite = true;
        for (float x : v) {
            finite = finite && std::isfinite(x);
        }
        if (!finite) {
            continue;
        }
        set->dirs[il] = std::move(v); // raw magnitude: scale 1 = one mean persona difference
    }
    set->ok = !set->dirs.empty();
    gguf_free(g);
    ggml_free(meta);
    return set;
}

// check = false skips the file-system check (used at compute time, after the graph build loaded it)
static const std::vector<float> * steer_lookup(const std::string & file, int il, bool check) {
    if (file.empty()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_steer_mutex);
    steer_set * last = nullptr;
    for (auto it = g_steer_sets.rbegin(); it != g_steer_sets.rend(); ++it) {
        if ((*it)->file == file) {
            last = it->get();
            break;
        }
    }
    if (!check && last != nullptr) {
        const auto it = last->dirs.find(il);
        return it == last->dirs.end() ? nullptr : &it->second;
    }
    long long stamp = 0;
    const bool exists = file_stamp(file, stamp);
    steer_set * cur = last;
    if (cur == nullptr || cur->exists != exists || cur->stamp != stamp) {
        g_steer_sets.push_back(load_steer(file, stamp, exists));
        cur = g_steer_sets.back().get();
    }
    const auto it = cur->dirs.find(il);
    return it == cur->dirs.end() ? nullptr : &it->second;
}

const std::vector<float> * llm_inj_steer_vector(const std::string & file, int il) {
    return steer_lookup(file, il, true);
}

// one warning per configuration generation
static std::atomic<uint64_t> g_steer_warned{UINT64_MAX};

static void steer_warn(const char * why) {
    const uint64_t gen = llm_inj_generation();
    if (g_steer_warned.exchange(gen) != gen) {
        std::fprintf(stderr, "llm-injection: ignoring LLM_INJ_STEER_FILE (%s)\n", why);
        std::fflush(stderr);
    }
}

// writes the steering direction of layer u.il (input: arange(0, n_embd))
static void op_steer(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) {
        return;
    }
    const unpacked u = unpack(ud);
    const std::vector<float> * v = steer_lookup(llm_inj_cfg().steer_file, u.il, false);
    const int64_t n = a->ne[0];
    float * out = (float *) dst->data;
    for (int64_t i = 0; i < n; ++i) {
        out[i] = (v != nullptr && (int64_t) v->size() == n) ? (*v)[i] : 0.0f;
    }
}

// ---------------------------------------------------------------------------------------------
// sites

float llm_inj_attn_scale(int il, int n_layer) {
    const llm_inj_config & c = llm_inj_cfg();
    if (c.attn_scale == 1.0f || !llm_inj_layer_in_range(c.attn_layers, il, n_layer)) {
        return 1.0f;
    }
    llm_inj_mark_fired(LLM_INJ_SITE_ATTENTION);
    const float m = llm_inj_pk_now();
    return 1.0f + (c.attn_scale - 1.0f) * m;
}

ggml_tensor * llm_inj_attn_heads(ggml_context * ctx, ggml_tensor * cur, int head_axis, int il, int n_layer) {
    const llm_inj_config & c = llm_inj_cfg();
    if (!c.heads_active() || cur == nullptr || cur->type != GGML_TYPE_F32 || head_axis < 0 || head_axis > 3 ||
        !llm_inj_layer_in_range(c.head_layers, il, n_layer)) {
        return cur;
    }
    const int64_t n_head = cur->ne[head_axis];
    if (n_head <= 0 || n_head > 65535) {
        return cur;
    }
    const float m = llm_inj_pk_now();
    ggml_tensor * idx  = ggml_arange(ctx, 0.0f, (float) n_head, 1.0f);
    ggml_tensor * mask = ggml_map_custom1(ctx, idx, op_mask, 1, pack((uint32_t) n_layer, il, OP_HEADS, m));
    int64_t ne[4] = { 1, 1, 1, 1 };
    ne[head_axis] = n_head;
    mask = ggml_reshape_4d(ctx, mask, ne[0], ne[1], ne[2], ne[3]);
    llm_inj_mark_fired(LLM_INJ_SITE_HEADS);
    return ggml_mul(ctx, cur, mask);
}

ggml_tensor * llm_inj_residual(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * prev, int il, int n_layer) {
    const llm_inj_config & c = llm_inj_cfg();
    if (il < 0 || cur == nullptr || cur->type != GGML_TYPE_F32) {
        return cur;
    }
    const float m = llm_inj_pk_now();

    // layer gain: rescale the block's contribution to the residual stream.
    // never applied to the first block (no input captured) or the last block (row-pruned outputs)
    if (c.layer_gain != 1.0f && prev != nullptr && il < n_layer - 1 && prev->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(prev, cur) && llm_inj_layer_in_range(c.gain_layers, il, n_layer)) {
        const float g = 1.0f + (c.layer_gain - 1.0f) * m;
        ggml_tensor * delta = ggml_sub(ctx, cur, prev);
        cur = ggml_add(ctx, prev, ggml_scale(ctx, delta, g));
        llm_inj_mark_fired(LLM_INJ_SITE_LAYER_GAIN);
    }

    // additive noise with RMS proportional to the token's own residual RMS: h' = h + a * rms(h) * u,
    // where u has unit RMS (sqrt(2) sin(...) in sin mode, a standard normal in hash mode)
    if (c.resid_noise > 0.0f && llm_inj_layer_in_range(c.resid_layers, il, n_layer)) {
        ggml_tensor * rms = ggml_sqrt(ctx, ggml_mean(ctx, ggml_sqr(ctx, cur))); // [1, n_tokens]
        ggml_tensor * u;
        if (c.noise_mode == LLM_INJ_NOISE_HASH) {
            u = ggml_map_custom1(ctx, cur, op_hash_elementwise, GGML_N_TASKS_MAX,
                                 pack((uint32_t) c.seed, il, OP_NOISE, 0.0f));
        } else {
            const float phase = layer_phase(c.seed, il, SALT_RESID);
            u = ggml_scale(ctx, ggml_sin(ctx, ggml_scale_bias(ctx, cur, K_FREQ_NOISE, phase)), 1.41421356f);
        }
        cur = ggml_add(ctx, cur, ggml_scale(ctx, ggml_mul(ctx, u, rms), c.resid_noise * m));
        llm_inj_mark_fired(LLM_INJ_SITE_RESID_NOISE);
    }

    // steering: add the layer's control vector (mean positive-minus-negative hidden state) to every token
    if (c.steer_active() && llm_inj_layer_in_range(c.steer_layers, il, n_layer)) {
        const std::vector<float> * v = llm_inj_steer_vector(c.steer_file, il);
        const int64_t n_embd = cur->ne[0];
        if (v == nullptr) {
            // control vectors have no direction for layer 0; only warn when nothing loaded at all
            bool any = false;
            for (int l = 0; l < n_layer && !any; ++l) {
                any = llm_inj_steer_vector(c.steer_file, l) != nullptr;
            }
            if (!any) {
                steer_warn("file missing or has no direction.<layer> tensors");
            }
        } else if ((int64_t) v->size() != n_embd) {
            steer_warn("direction size does not match the model's embedding size");
        } else {
            ggml_tensor * idx = ggml_arange(ctx, 0.0f, (float) n_embd, 1.0f);
            ggml_tensor * dir = ggml_map_custom1(ctx, idx, op_steer, 1, pack(0, il, OP_STEER, 0.0f));
            cur = ggml_add(ctx, cur, ggml_scale(ctx, dir, c.steer_scale * m)); // broadcasts over tokens
            llm_inj_mark_fired(LLM_INJ_SITE_STEER);
        }
    }

    return cur;
}

ggml_tensor * llm_inj_ffn(ggml_context * ctx, ggml_tensor * cur, int il, int n_layer) {
    const llm_inj_config & c = llm_inj_cfg();
    if ((c.ffn_dropout <= 0.0f && c.ffn_lesion <= 0.0f) || il < 0 || cur == nullptr || cur->type != GGML_TYPE_F32 ||
        !llm_inj_layer_in_range(c.ffn_layers, il, n_layer)) {
        return cur;
    }
    const float m = llm_inj_pk_now();

    // fixed channel lesion: the same output channels are silenced for every token
    if (c.ffn_lesion > 0.0f) {
        ggml_tensor * idx  = ggml_arange(ctx, 0.0f, (float) cur->ne[0], 1.0f);
        ggml_tensor * mask = ggml_map_custom1(ctx, idx, op_mask, 1, pack(0, il, OP_FFN_LESION, m));
        cur = ggml_mul(ctx, cur, mask);
        llm_inj_mark_fired(LLM_INJ_SITE_FFN_LESION);
    }

    // transient dropout: a different subset of units for every activation pattern
    if (c.ffn_dropout > 0.0f) {
        const float p = std::min(c.ffn_dropout * m, 0.95f);
        if (c.noise_mode == LLM_INJ_NOISE_HASH) {
            ggml_tensor * keep = ggml_map_custom1(ctx, cur, op_hash_elementwise, GGML_N_TASKS_MAX,
                                                  pack((uint32_t) c.seed, il, OP_DROPOUT, p));
            cur = ggml_mul(ctx, cur, keep);
        } else {
            // keep unit iff sin(K x + phase) > t, with t chosen so that P(keep) = 1 - p for a uniform phase:
            //   P(sin(U) > t) = 1/2 - asin(t)/pi  =>  t = -cos(pi * p)
            const float t     = -std::cos(3.14159265358979f * p);
            const float phase = layer_phase(c.seed, il, SALT_DROPOUT);
            ggml_tensor * u    = ggml_sin(ctx, ggml_scale_bias(ctx, cur, K_FREQ_DROPOUT, phase));
            ggml_tensor * keep = ggml_step(ctx, ggml_scale_bias(ctx, u, 1.0f, -t));
            cur = ggml_scale(ctx, ggml_mul(ctx, cur, keep), 1.0f / (1.0f - p));
        }
        llm_inj_mark_fired(LLM_INJ_SITE_FFN_DROPOUT);
    }
    return cur;
}
