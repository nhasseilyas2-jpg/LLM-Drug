// llm-injection core: configuration, hashing, logits site. No ggml dependency (unit-testable).
// Part of LLM Injection Runtime Lab (MIT License).

#include "llm-injection.h"

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>

// ---------------------------------------------------------------------------------------------
// hashing

uint64_t llm_inj_hash(uint64_t a, uint64_t b) {
    // splitmix64 finalizer over a combined key
    uint64_t z = a + 0x9e3779b97f4a7c15ULL * (b + 0x632be59bd9b4e019ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

double llm_inj_unit(uint64_t h) {
    return (double) (h >> 11) * (1.0 / 9007199254740992.0); // 2^-53
}

double llm_inj_gauss(uint64_t h) {
    // Box-Muller with both uniforms taken from one 64-bit hash
    const double u1 = ((double) (h >> 32) + 1.0) * (1.0 / 4294967296.0);  // (0, 1]
    const double u2 = (double) (h & 0xffffffffULL) * (1.0 / 4294967296.0); // [0, 1)
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

// ---------------------------------------------------------------------------------------------
// configuration

bool llm_inj_config::logits_active() const {
    return logit_noise > 0.0f || logit_temp != 1.0f || top_suppress > 0.0f ||
           (tail_boost > 0.0f && tail_count > 0) || (fixation_bias != 0.0f && !fixation_ids.empty());
}

bool llm_inj_config::graph_active() const {
    return attn_scale != 1.0f || resid_noise > 0.0f || layer_gain != 1.0f || ffn_dropout > 0.0f;
}

static const char * env_str(const char * name) {
    const char * v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? v : nullptr;
}

static void warn_invalid(const char * name, const char * value) {
    std::fprintf(stderr, "llm-injection: ignoring invalid %s='%s'\n", name, value);
}

static float env_float(const char * name, float def, float lo, float hi) {
    const char * v = env_str(name);
    if (!v) {
        return def;
    }
    char * end = nullptr;
    const double x = std::strtod(v, &end);
    if (end == v || !std::isfinite(x)) {
        warn_invalid(name, v);
        return def;
    }
    return (float) std::min<double>(hi, std::max<double>(lo, x));
}

static int64_t env_int(const char * name, int64_t def, int64_t lo, int64_t hi) {
    const char * v = env_str(name);
    if (!v) {
        return def;
    }
    char * end = nullptr;
    const long long x = std::strtoll(v, &end, 10);
    if (end == v) {
        warn_invalid(name, v);
        return def;
    }
    return std::min<int64_t>(hi, std::max<int64_t>(lo, (int64_t) x));
}

static uint64_t env_u64(const char * name, uint64_t def) {
    const char * v = env_str(name);
    if (!v) {
        return def;
    }
    char * end = nullptr;
    const unsigned long long x = std::strtoull(v, &end, 10);
    if (end == v) {
        warn_invalid(name, v);
        return def;
    }
    return (uint64_t) x;
}

bool llm_inj_parse_range(const char * text, llm_inj_range & out) {
    if (text == nullptr) {
        return false;
    }
    std::string s(text);
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }), s.end());
    const size_t colon = s.find(':');
    if (colon == std::string::npos) {
        return false;
    }
    std::string a = s.substr(0, colon);
    std::string b = s.substr(colon + 1);
    const bool abs_a = !a.empty() && (a[0] == 'L' || a[0] == 'l');
    const bool abs_b = !b.empty() && (b[0] == 'L' || b[0] == 'l');
    if (abs_a != abs_b) {
        return false;
    }
    if (abs_a) {
        a = a.substr(1);
        b = b.substr(1);
    }
    if (a.empty() || b.empty()) {
        return false;
    }
    char * end_a = nullptr;
    char * end_b = nullptr;
    const double lo = std::strtod(a.c_str(), &end_a);
    const double hi = std::strtod(b.c_str(), &end_b);
    if (*end_a != '\0' || *end_b != '\0' || !std::isfinite(lo) || !std::isfinite(hi) || lo > hi || lo < 0.0) {
        return false;
    }
    if (!abs_a && hi > 1.0) {
        return false;
    }
    if (abs_a && (lo != std::floor(lo) || hi != std::floor(hi))) {
        return false;
    }
    out.absolute = abs_a;
    out.lo       = (float) lo;
    out.hi       = (float) hi;
    return true;
}

static llm_inj_range env_range(const char * name) {
    llm_inj_range r;
    const char * v = env_str(name);
    if (v && !llm_inj_parse_range(v, r)) {
        warn_invalid(name, v);
        r = llm_inj_range();
    }
    return r;
}

static std::vector<int32_t> env_ids(const char * name) {
    std::vector<int32_t> ids;
    const char * v = env_str(name);
    if (!v) {
        return ids;
    }
    const char * p = v;
    while (*p) {
        char * end = nullptr;
        const long x = std::strtol(p, &end, 10);
        if (end == p) {
            ++p; // skip separators
            continue;
        }
        if (x >= 0 && x <= INT32_MAX && ids.size() < 4096) {
            ids.push_back((int32_t) x);
        }
        p = end;
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

llm_inj_config llm_inj_parse_env() {
    llm_inj_config c;
    c.seed          = env_u64  ("LLM_INJ_SEED", 0);
    c.log           = env_int  ("LLM_INJ_LOG", 0, 0, 1) != 0;

    c.logit_noise   = env_float("LLM_INJ_LOGIT_NOISE",   0.0f, 0.0f,   50.0f);
    c.logit_temp    = env_float("LLM_INJ_LOGIT_TEMP",    1.0f, 0.05f,  20.0f);
    c.top_suppress  = env_float("LLM_INJ_TOP_SUPPRESS",  0.0f, 0.0f,    1.0f);
    c.tail_boost    = env_float("LLM_INJ_TAIL_BOOST",    0.0f, 0.0f,  100.0f);
    c.tail_count    = (int) env_int("LLM_INJ_TAIL_COUNT", 32, 0, 4096);
    c.fixation_bias = env_float("LLM_INJ_FIXATION_BIAS", 0.0f, -100.0f, 100.0f);
    c.fixation_ids  = env_ids  ("LLM_INJ_FIXATION_IDS");

    c.attn_scale    = env_float("LLM_INJ_ATTN_SCALE",    1.0f, 0.01f,  20.0f);
    c.attn_layers   = env_range("LLM_INJ_ATTN_LAYERS");

    c.resid_noise   = env_float("LLM_INJ_RESID_NOISE",   0.0f, 0.0f,   10.0f);
    c.resid_layers  = env_range("LLM_INJ_RESID_LAYERS");
    c.layer_gain    = env_float("LLM_INJ_LAYER_GAIN",    1.0f, -4.0f,   4.0f);
    c.gain_layers   = env_range("LLM_INJ_GAIN_LAYERS");

    c.ffn_dropout   = env_float("LLM_INJ_FFN_DROPOUT",   0.0f, 0.0f,    0.95f);
    c.ffn_layers    = env_range("LLM_INJ_FFN_LAYERS");

    c.kv_forget     = env_float("LLM_INJ_KV_FORGET",     0.0f, 0.0f,    1.0f);
    c.kv_recent     = (int32_t) env_int("LLM_INJ_KV_RECENT", 32, 0, 1 << 24);
    c.kv_sink       = (int32_t) env_int("LLM_INJ_KV_SINK",    4, 0, 1 << 24);
    return c;
}

static std::string range_str(const llm_inj_range & r) {
    char buf[64];
    if (r.absolute) {
        std::snprintf(buf, sizeof(buf), "L%d:L%d", (int) r.lo, (int) r.hi);
    } else {
        std::snprintf(buf, sizeof(buf), "%.3g:%.3g", r.lo, r.hi);
    }
    return buf;
}

std::string llm_inj_describe(const llm_inj_config & c) {
    std::ostringstream os;
    os << "seed=" << c.seed;
    if (c.logits_active()) {
        os << " logits{noise=" << c.logit_noise << " temp=" << c.logit_temp << " top_suppress=" << c.top_suppress
           << " tail=" << c.tail_boost << "x" << c.tail_count << " fixation=" << c.fixation_bias << "x"
           << c.fixation_ids.size() << "}";
    }
    if (c.attn_scale != 1.0f) {
        os << " attention{scale=" << c.attn_scale << " layers=" << range_str(c.attn_layers) << "}";
    }
    if (c.resid_noise > 0.0f) {
        os << " resid_noise{rel=" << c.resid_noise << " layers=" << range_str(c.resid_layers) << "}";
    }
    if (c.layer_gain != 1.0f) {
        os << " layer_gain{g=" << c.layer_gain << " layers=" << range_str(c.gain_layers) << "}";
    }
    if (c.ffn_dropout > 0.0f) {
        os << " ffn_dropout{p=" << c.ffn_dropout << " layers=" << range_str(c.ffn_layers) << "}";
    }
    if (c.kv_active()) {
        os << " kv_forget{p=" << c.kv_forget << " recent=" << c.kv_recent << " sink=" << c.kv_sink << "}";
    }
    return os.str();
}

const llm_inj_config & llm_inj_cfg() {
    static const llm_inj_config cfg = [] {
        llm_inj_config c = llm_inj_parse_env();
        if (c.any_active()) {
            std::fprintf(stderr, "llm-injection: ACTIVE %s\n", llm_inj_describe(c).c_str());
        } else if (c.log) {
            std::fprintf(stderr, "llm-injection: inactive (no LLM_INJ_* effect configured)\n");
        }
        return c;
    }();
    return cfg;
}

static std::atomic<bool> g_fired[LLM_INJ_SITE_COUNT];

const char * llm_inj_site_name(llm_inj_site site) {
    switch (site) {
        case LLM_INJ_SITE_LOGITS:      return "logits";
        case LLM_INJ_SITE_ATTENTION:   return "attention";
        case LLM_INJ_SITE_RESID_NOISE: return "resid_noise";
        case LLM_INJ_SITE_LAYER_GAIN:  return "layer_gain";
        case LLM_INJ_SITE_FFN_DROPOUT: return "ffn_dropout";
        case LLM_INJ_SITE_KV_FORGET:   return "kv_forget";
        default:                       return "unknown";
    }
}

void llm_inj_mark_fired(llm_inj_site site) {
    if (site < 0 || site >= LLM_INJ_SITE_COUNT) {
        return;
    }
    if (g_fired[site].load(std::memory_order_relaxed)) {
        return;
    }
    if (!g_fired[site].exchange(true)) {
        std::fprintf(stderr, "llm-injection: site fired: %s\n", llm_inj_site_name(site));
        std::fflush(stderr);
    }
}

bool llm_inj_has_fired(llm_inj_site site) {
    return site >= 0 && site < LLM_INJ_SITE_COUNT && g_fired[site].load(std::memory_order_relaxed);
}

bool llm_inj_layer_in_range(const llm_inj_range & r, int il, int n_layer) {
    if (il < 0 || n_layer <= 0 || il >= n_layer) {
        return false;
    }
    if (r.absolute) {
        return (float) il >= r.lo && (float) il <= r.hi;
    }
    const float depth = ((float) il + 0.5f) / (float) n_layer;
    return depth >= r.lo && depth <= r.hi;
}

// ---------------------------------------------------------------------------------------------
// logits site

static const uint64_t K_NOISE    = 0x6e6f697365000001ULL;
static const uint64_t K_TAIL     = 0x7461696c00000002ULL;
static const uint64_t K_SUPPRESS = 0x7375707072000003ULL;

void llm_inj_apply_logits_cfg(const llm_inj_config & c, llama_token_data_array * cur_p,
                              uint64_t chain_seed, uint64_t step) {
    if (cur_p == nullptr || cur_p->data == nullptr || cur_p->size == 0 || !c.logits_active()) {
        return;
    }

    llama_token_data * d = cur_p->data;
    const size_t       n = cur_p->size;

    // statistics of the finite logits (masked tokens stay masked)
    double sum = 0.0, sum2 = 0.0;
    size_t nf  = 0;
    for (size_t i = 0; i < n; ++i) {
        const float l = d[i].logit;
        if (std::isfinite(l)) {
            sum  += l;
            sum2 += (double) l * l;
            ++nf;
        }
    }
    if (nf == 0) {
        return;
    }
    const double mean = sum / (double) nf;
    const double var  = sum2 / (double) nf - mean * mean;
    const float  sd   = var > 1e-12 ? (float) std::sqrt(var) : 1.0f;

    const uint64_t base = llm_inj_hash(llm_inj_hash(c.seed, chain_seed), step);

    // 1) gaussian noise, keyed by token id so it does not depend on candidate order
    if (c.logit_noise > 0.0f) {
        const float amp = c.logit_noise * sd;
        const uint64_t kn = llm_inj_hash(base, K_NOISE);
        for (size_t i = 0; i < n; ++i) {
            if (std::isfinite(d[i].logit)) {
                d[i].logit += amp * (float) llm_inj_gauss(llm_inj_hash(kn, (uint64_t) (uint32_t) d[i].id));
            }
        }
    }

    // 2) tail boost: promote random candidates (re-drawn every step)
    if (c.tail_boost > 0.0f && c.tail_count > 0) {
        const float amp = c.tail_boost * sd;
        const uint64_t kt = llm_inj_hash(base, K_TAIL);
        for (int k = 0; k < c.tail_count; ++k) {
            const size_t idx = (size_t) (llm_inj_hash(kt, (uint64_t) k) % (uint64_t) n);
            if (std::isfinite(d[idx].logit)) {
                d[idx].logit += amp;
            }
        }
    }

    // 3) fixation: persistent bias toward a fixed token set
    if (c.fixation_bias != 0.0f && !c.fixation_ids.empty()) {
        const float amp = c.fixation_bias * sd;
        for (const int32_t id : c.fixation_ids) {
            if ((size_t) id < n && d[id].id == id) {
                // fast path: unsorted full-vocabulary array
                if (std::isfinite(d[id].logit)) {
                    d[id].logit += amp;
                }
                continue;
            }
            for (size_t i = 0; i < n; ++i) {
                if (d[i].id == id) {
                    if (std::isfinite(d[i].logit)) {
                        d[i].logit += amp;
                    }
                    break;
                }
            }
        }
    }

    // 4) top-1 suppression: veto the currently preferred token (never the last finite one)
    if (c.top_suppress > 0.0f && nf >= 2 &&
        llm_inj_unit(llm_inj_hash(base, K_SUPPRESS)) < (double) c.top_suppress) {
        size_t imax = n;
        float  lmax = -std::numeric_limits<float>::infinity();
        float  lmin =  std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < n; ++i) {
            const float l = d[i].logit;
            if (std::isfinite(l)) {
                if (l > lmax) { lmax = l; imax = i; }
                if (l < lmin) { lmin = l; }
            }
        }
        if (imax < n) {
            // stays finite so grammar-constrained sampling can still fall back to it
            d[imax].logit = lmin - 4.0f * sd;
        }
    }

    // 5) temperature around the mean (softmax is shift invariant)
    if (c.logit_temp != 1.0f) {
        const float inv_t = 1.0f / c.logit_temp;
        const float m     = (float) mean;
        for (size_t i = 0; i < n; ++i) {
            if (std::isfinite(d[i].logit)) {
                d[i].logit = m + (d[i].logit - m) * inv_t;
            }
        }
    }

    cur_p->sorted = false;
    llm_inj_mark_fired(LLM_INJ_SITE_LOGITS);
}

void llm_inj_apply_logits(llama_token_data_array * cur_p, uint64_t chain_seed, uint64_t step) {
    const llm_inj_config & c = llm_inj_cfg();
    if (!c.logits_active()) {
        return;
    }
    llm_inj_apply_logits_cfg(c, cur_p, chain_seed, step);
}
