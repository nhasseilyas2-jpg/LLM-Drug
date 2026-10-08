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
//   kv          | KQ mask (hides cached positions)        | LLM_INJ_KV_FORGET, LLM_INJ_KV_RECENT, LLM_INJ_KV_SINK
//
// Common: LLM_INJ_SEED (uint64), LLM_INJ_LOG (1 = verbose config dump).
// Layer ranges are fractional depth "0.25:0.75" or absolute inclusive indices "L3:L10".
//
// See llamacpp-injection/README.md for the exact math of every site.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct llama_token_data_array;

enum llm_inj_site {
    LLM_INJ_SITE_LOGITS = 0,
    LLM_INJ_SITE_ATTENTION,
    LLM_INJ_SITE_RESID_NOISE,
    LLM_INJ_SITE_LAYER_GAIN,
    LLM_INJ_SITE_FFN_DROPOUT,
    LLM_INJ_SITE_KV_FORGET,
    LLM_INJ_SITE_COUNT,
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

    // feed-forward
    float         ffn_dropout = 0.0f; // fraction of FFN output units zeroed (inverted dropout)
    llm_inj_range ffn_layers;

    // KV memory
    float   kv_forget = 0.0f; // probability that a cached position is hidden from attention
    int32_t kv_recent = 32;   // positions within this distance of the query are never hidden
    int32_t kv_sink   = 4;    // the first kv_sink positions (attention sinks) are never hidden

    bool logits_active() const;
    bool graph_active()  const;
    bool kv_active()     const { return kv_forget > 0.0f; }
    bool any_active()    const { return logits_active() || graph_active() || kv_active(); }
};

// ---------------------------------------------------------------------------------------------
// configuration

// parse the current environment (pure; used by tests). Invalid values fall back to defaults.
llm_inj_config llm_inj_parse_env();

// parse a single layer range string ("0.25:0.75" or "L3:L10"); returns false if invalid
bool llm_inj_parse_range(const char * text, llm_inj_range & out);

// process-wide configuration, parsed once from the environment (thread-safe)
const llm_inj_config & llm_inj_cfg();

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

// ---------------------------------------------------------------------------------------------
// kv site: true if the cached position p0 must be hidden from the query at position p1

inline bool llm_inj_kv_forget_pos(const llm_inj_config & cfg, int32_t p0, int32_t p1) {
    if (p0 < cfg.kv_sink || (p1 - p0) <= cfg.kv_recent) {
        return false;
    }
    return llm_inj_unit(llm_inj_hash(cfg.seed ^ 0x6b762d666f726765ULL, (uint64_t) (uint32_t) p0)) < cfg.kv_forget;
}

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
