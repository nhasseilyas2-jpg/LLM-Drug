// llm-injection: runtime perturbation ("injection") engine for llama.cpp.
//
// Part of LLM Injection Runtime Lab (MIT License).
//
// Every intervention is configured once per process from LLM_INJ_* environment
// variables and is a strict no-op when nothing is configured. Model weights are
// never modified; perturbations are applied to the computation at run time:
//
//   site        | where                                   | knobs
//   ------------+-----------------------------------------+------------------------------------------
//   logits      | start of the sampler chain (per token)  | LLM_INJ_LOGIT_NOISE / _TEMP / _TOP_SUPPRESS
//               |                                         | LLM_INJ_TAIL_BOOST / _TAIL_COUNT
//               |                                         | LLM_INJ_FIXATION_BIAS / _FIXATION_IDS
//   attention   | softmax scale of QK^T (build_attn_mha)  | LLM_INJ_ATTN_SCALE, LLM_INJ_ATTN_LAYERS
//   residual    | end of each block (build_cvec)          | LLM_INJ_RESID_NOISE, LLM_INJ_RESID_LAYERS
//               |                                         | LLM_INJ_LAYER_GAIN,  LLM_INJ_GAIN_LAYERS
//   ffn         | output of dense / MoE FFN               | LLM_INJ_FFN_DROPOUT, LLM_INJ_FFN_LAYERS
//   heads       | per-head attention output               | LLM_INJ_HEAD_LESION, LLM_INJ_HEAD_GAIN,
//               |                                         | LLM_INJ_HEAD_IDS, LLM_INJ_HEAD_LAYERS
//   ffn         | output of dense / MoE FFN               | LLM_INJ_FFN_DROPOUT (transient), LLM_INJ_FFN_LESION
//               |                                         | (fixed channels), LLM_INJ_FFN_LAYERS
//   kv          | KQ mask (hides cached positions)        | LLM_INJ_KV_FORGET, LLM_INJ_KV_RECENT, LLM_INJ_KV_SINK
//   steer       | end of each block (build_cvec)          | LLM_INJ_STEER_FILE, LLM_INJ_STEER_SCALE,
//               |                                         | LLM_INJ_STEER_LAYERS
//
// Common: LLM_INJ_SEED (uint64), LLM_INJ_LOG (1 = verbose config dump),
//         LLM_INJ_NOISE_MODE (sin | hash), LLM_INJ_PK_ONSET / LLM_INJ_PK_HALFLIFE (time course, tokens),
//         LLM_INJ_CONFIG_FILE (KEY=VALUE file re-read at the start of every request).
// Layer ranges are fractional depth "0.25:0.75" or absolute inclusive indices "L3:L10".
//
// See llamacpp-injection/README.md for the exact math of every site.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

struct llama_token_data_array;

enum llm_inj_site {
    LLM_INJ_SITE_LOGITS = 0,
    LLM_INJ_SITE_ATTENTION,
    LLM_INJ_SITE_RESID_NOISE,
    LLM_INJ_SITE_LAYER_GAIN,
    LLM_INJ_SITE_FFN_DROPOUT,
    LLM_INJ_SITE_KV_FORGET,
    LLM_INJ_SITE_HEADS,
    LLM_INJ_SITE_FFN_LESION,
    LLM_INJ_SITE_STEER,
    LLM_INJ_SITE_COUNT,
};

enum llm_inj_noise_mode {
    LLM_INJ_NOISE_SIN  = 0, // sin(K x + phase): pure ggml ops, runs on every backend (default)
    LLM_INJ_NOISE_HASH = 1, // counter hash of (seed, layer, channel, value bits): CPU custom op
};

struct llm_inj_range {
    bool  absolute = false; // true: [lo, hi] are inclusive layer indices; false: fractional depth
    float lo       = 0.0f;
    float hi       = 1.0f;
};

struct llm_inj_config {
    uint64_t seed = 0;
    bool     log  = false;

    // logits (sampler chain)
    float logit_noise   = 0.0f; // gaussian noise, in units of the per-step std of the finite logits
    float logit_temp    = 1.0f; // extra temperature applied to the logits (>1 flattens, <1 sharpens)
    float top_suppress  = 0.0f; // per-step probability of vetoing the current top-1 token
    float tail_boost    = 0.0f; // boost (std units) added to `tail_count` random tokens, re-drawn every step
    int   tail_count    = 32;
    float fixation_bias = 0.0f; // persistent bias (std units) on `fixation_ids`
    std::vector<int32_t> fixation_ids;

    // attention
    float         attn_scale = 1.0f; // multiplier on the QK^T softmax scale (<1 diffuse, >1 sharp)
    llm_inj_range attn_layers;

    // residual stream
    float         resid_noise = 0.0f; // additive pseudo-noise, RMS = resid_noise * RMS(token's residual)
    llm_inj_range resid_layers;
    float         layer_gain  = 1.0f; // h_out = h_in + g * (block(h_in) - h_in); 0 = skip the block
    llm_inj_range gain_layers;

    // attention heads ("lesions"): the selected heads' outputs are multiplied by head_gain
    float         head_lesion = 0.0f; // fraction of heads per layer selected pseudo-randomly (by seed)
    float         head_gain   = 0.0f; // 0 = ablate, 0.5 = attenuate, 2 = amplify
    std::vector<std::pair<int32_t, int32_t>> head_ids; // explicit (layer, head) pairs, e.g. "3:5,10:0"
    llm_inj_range head_layers;

    // feed-forward
    float         ffn_dropout = 0.0f; // fraction of FFN output units zeroed (inverted dropout)
    float         ffn_lesion  = 0.0f; // fraction of FFN output channels silenced for every token
    llm_inj_range ffn_layers;

    // KV memory
    float   kv_forget = 0.0f; // probability that a cached position is hidden from attention
    int32_t kv_recent = 32;   // positions within this distance of the query are never hidden
    int32_t kv_sink   = 4;    // the first kv_sink positions (attention sinks) are never hidden

    // graph-site pseudo-noise generator
    int noise_mode = LLM_INJ_NOISE_SIN;

    // pharmacokinetics: every effect is scaled by m(t), t = tokens generated in the current request
    //   m(t) = (1 - exp(-ln(20) t / onset)) * 0.5^(max(0, t - onset) / halflife)
    // onset = 0: immediate (absorption term = 1); halflife = 0: no elimination
    float pk_onset    = 0.0f;
    float pk_halflife = 0.0f;

    // activation steering: h' = h + steer_scale * v[il], where v[il] is the control vector (mean
    // positive-minus-negative hidden state) for layer il read from a control-vector GGUF (tensors "direction.<il>", as written by
    // llama-cvector-generator). Negative scales push the other way.
    std::string   steer_file;
    float         steer_scale = 0.0f;
    llm_inj_range steer_layers;

    bool heads_active()  const { return (head_lesion > 0.0f || !head_ids.empty()) && head_gain != 1.0f; }
    bool steer_active()  const { return steer_scale != 0.0f && !steer_file.empty(); }
    bool pk_active()     const { return pk_onset > 0.0f || pk_halflife > 0.0f; }
    bool logits_active() const;
    bool graph_active()  const;
    bool kv_active()     const { return kv_forget > 0.0f; }
    bool any_active()    const { return logits_active() || graph_active() || kv_active(); }
};

// ---------------------------------------------------------------------------------------------
// configuration

// returns the raw value of a key (nullptr if unset). Used to parse env, files and test maps.
typedef std::function<const char *(const char *)> llm_inj_getter;

// parse a configuration from any key source. Invalid values fall back to defaults with a warning.
llm_inj_config llm_inj_parse(const llm_inj_getter & get);

// parse the current environment (pure; used by tests).
llm_inj_config llm_inj_parse_env();

// parse KEY=VALUE lines (blank lines and '#' comments ignored); unknown keys fall back to `fallback`
llm_inj_config llm_inj_parse_text(const std::string & text, const llm_inj_getter & fallback);

// parse a single layer range string ("0.25:0.75" or "L3:L10"); returns false if invalid
bool llm_inj_parse_range(const char * text, llm_inj_range & out);

// parse "layer:head" pairs separated by ',' / ';' / whitespace. Invalid items are skipped.
std::vector<std::pair<int32_t, int32_t>> llm_inj_parse_heads(const char * text);

// process-wide configuration. Parsed from the environment on first use; if LLM_INJ_CONFIG_FILE is
// set, it is re-read by llm_inj_begin_sequence() and replaced when its content changes.
const llm_inj_config & llm_inj_cfg();

// configuration generation: incremented every time the configuration is replaced
uint64_t llm_inj_generation();

// called when a new sampler chain is created (start of a request): reloads the config file if
// configured and resets the token clock used by the pharmacokinetic schedule
void llm_inj_begin_sequence();

// token clock (tokens sampled in the current request)
uint64_t llm_inj_step();

// pharmacokinetic factor m(t) in [0, 1] for a configuration (1 when no schedule is configured)
float llm_inj_pk_factor(const llm_inj_config & cfg, double t);

// pharmacokinetic factor at the current token clock
float llm_inj_pk_now();

// copy of cfg with every effect scaled by m (m = 1 returns an identical configuration)
llm_inj_config llm_inj_scaled(const llm_inj_config & cfg, float m);

// graph reuse guard: false when the graph must be rebuilt because the configuration changed or a
// time-varying graph-site schedule is active. llm_inj_graph_built() records the state of a build.
bool llm_inj_graph_reusable();
void llm_inj_graph_built();

std::string llm_inj_describe(const llm_inj_config & cfg);

// record that an injection site executed; prints a one-time audit line to stderr
void llm_inj_mark_fired(llm_inj_site site);
bool llm_inj_has_fired(llm_inj_site site);
const char * llm_inj_site_name(llm_inj_site site);

// ---------------------------------------------------------------------------------------------
// helpers

bool llm_inj_layer_in_range(const llm_inj_range & r, int il, int n_layer);

// counter-based hashing (splitmix64) -> reproducible randomness without shared RNG state
uint64_t llm_inj_hash(uint64_t a, uint64_t b);
double   llm_inj_unit(uint64_t h);   // [0, 1)
double   llm_inj_gauss(uint64_t h);  // standard normal

// ---------------------------------------------------------------------------------------------
// logits site: perturb candidate logits before the sampler chain runs.
// chain_seed: seed of the chain's dist sampler; step: per-chain call counter.

void llm_inj_apply_logits_cfg(const llm_inj_config & cfg, llama_token_data_array * cur_p,
                              uint64_t chain_seed, uint64_t step);
void llm_inj_apply_logits(llama_token_data_array * cur_p, uint64_t chain_seed, uint64_t step);

// called by the sampler chain for every token: advances the token clock and applies the logits
// site with the pharmacokinetic factor of this step
void llm_inj_on_sample(llama_token_data_array * cur_p, uint64_t chain_seed, uint64_t step);

// ---------------------------------------------------------------------------------------------
// kv site: true if the cached position p0 must be hidden from the query at position p1.
// p_forget is the effective forgetting probability (cfg.kv_forget scaled by the schedule).
// The hidden set is nested: a position hidden at probability p stays hidden for every p' > p.

inline bool llm_inj_kv_forget_pos(const llm_inj_config & cfg, float p_forget, int32_t p0, int32_t p1) {
    if (p_forget <= 0.0f || p0 < cfg.kv_sink || (p1 - p0) <= cfg.kv_recent) {
        return false;
    }
    return llm_inj_unit(llm_inj_hash(cfg.seed ^ 0x6b762d666f726765ULL, (uint64_t) (uint32_t) p0)) < p_forget;
}

inline bool llm_inj_kv_forget_pos(const llm_inj_config & cfg, int32_t p0, int32_t p1) {
    return llm_inj_kv_forget_pos(cfg, cfg.kv_forget, p0, p1);
}

// ---------------------------------------------------------------------------------------------
// lesion selection (pure; shared by the graph ops and the unit tests)

// gain applied to head `head` of layer il (1 = untouched). frac and the gain change are scaled by m.
// Random lesions: the round(frac * n_head) heads with the smallest hash(seed, il, head) are selected,
// so the selected set grows monotonically with the dose.
float llm_inj_head_gain(const llm_inj_config & cfg, float m, int il, int n_layer, int head, int n_head);

// true if FFN output channel c of layer il is silenced (fixed for all tokens; nested in the fraction)
bool llm_inj_ffn_lesioned(const llm_inj_config & cfg, float m, int il, int64_t c);

// hash-mode noise value (standard normal) for one activation, keyed by value bits and channel
float llm_inj_hash_noise(uint64_t seed, int il, uint64_t salt, int64_t channel, float x);

// ---------------------------------------------------------------------------------------------
// graph sites (implemented in llm-injection-graph.cpp, require ggml)

struct ggml_context;
struct ggml_tensor;

// attention: multiplier for kq_scale of layer il
float llm_inj_attn_scale(int il, int n_layer);

// residual stream at the end of block il. prev = residual stream entering block il (or nullptr)
ggml_tensor * llm_inj_residual(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * prev, int il, int n_layer);

// output of the (dense or MoE) feed-forward block of layer il
ggml_tensor * llm_inj_ffn(ggml_context * ctx, ggml_tensor * cur, int il, int n_layer);

// per-head attention output of layer il; head_axis is the dimension that indexes heads
ggml_tensor * llm_inj_attn_heads(ggml_context * ctx, ggml_tensor * cur, int head_axis, int il, int n_layer);

// steering directions loaded from a control-vector GGUF: raw vector for layer il, or nullptr
// (file missing/unreadable, no direction for that layer). Cached per file path and modification time.
const std::vector<float> * llm_inj_steer_vector(const std::string & file, int il);
