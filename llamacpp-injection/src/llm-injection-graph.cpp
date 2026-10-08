// llm-injection graph sites: attention temperature, residual-stream noise / layer gain, FFN dropout.
// Part of LLM Injection Runtime Lab (MIT License).
//
// All ops are plain ggml element-wise ops, so they run on any backend that supports
// sin/step/sqrt/mean (CPU, CUDA, Metal, Vulkan).
//
// Pseudo-randomness inside the graph: we cannot draw RNG samples per element inside a ggml graph,
// so noise and dropout masks are derived from the activations themselves:
//     u(x) = sin(K * x + phase_layer)
// With a large K, u(x) behaves like a zero-mean, bounded, high-frequency hash of each activation
// (RMS 1/sqrt(2)); the per-layer phase comes from LLM_INJ_SEED, so different seeds give
// different (but reproducible) perturbations.

#include "llm-injection.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>

static const float K_FREQ_NOISE   = 4099.0f;
static const float K_FREQ_DROPOUT = 3571.0f;

static float layer_phase(uint64_t seed, int il, uint64_t salt) {
    return (float) (6.283185307179586 * llm_inj_unit(llm_inj_hash(llm_inj_hash(seed, salt), (uint64_t) (uint32_t) il)));
}

float llm_inj_attn_scale(int il, int n_layer) {
    const llm_inj_config & c = llm_inj_cfg();
    if (c.attn_scale == 1.0f || !llm_inj_layer_in_range(c.attn_layers, il, n_layer)) {
        return 1.0f;
    }
    llm_inj_mark_fired(LLM_INJ_SITE_ATTENTION);
    return c.attn_scale;
}

ggml_tensor * llm_inj_residual(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * prev, int il, int n_layer) {
    const llm_inj_config & c = llm_inj_cfg();
    if (il < 0 || cur == nullptr || cur->type != GGML_TYPE_F32) {
        return cur;
    }

    // layer gain: rescale the block's contribution to the residual stream.
    // never applied to the first block (no input captured) or the last block (row-pruned outputs)
    if (c.layer_gain != 1.0f && prev != nullptr && il < n_layer - 1 && prev->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(prev, cur) && llm_inj_layer_in_range(c.gain_layers, il, n_layer)) {
        ggml_tensor * delta = ggml_sub(ctx, cur, prev);
        cur = ggml_add(ctx, prev, ggml_scale(ctx, delta, c.layer_gain));
        llm_inj_mark_fired(LLM_INJ_SITE_LAYER_GAIN);
    }

    // additive noise with RMS proportional to the token's own residual RMS:
    //   h' = h + a * rms(h) * sqrt(2) * sin(K h + phase)
    if (c.resid_noise > 0.0f && llm_inj_layer_in_range(c.resid_layers, il, n_layer)) {
        const float phase = layer_phase(c.seed, il, 0x7265736964ULL);
        ggml_tensor * rms   = ggml_sqrt(ctx, ggml_mean(ctx, ggml_sqr(ctx, cur)));          // [1, n_tokens]
        ggml_tensor * u     = ggml_sin(ctx, ggml_scale_bias(ctx, cur, K_FREQ_NOISE, phase)); // [n_embd, n_tokens]
        ggml_tensor * noise = ggml_mul(ctx, u, rms);
        cur = ggml_add(ctx, cur, ggml_scale(ctx, noise, c.resid_noise * 1.41421356f));
        llm_inj_mark_fired(LLM_INJ_SITE_RESID_NOISE);
    }

    return cur;
}

ggml_tensor * llm_inj_ffn(ggml_context * ctx, ggml_tensor * cur, int il, int n_layer) {
    const llm_inj_config & c = llm_inj_cfg();
    if (c.ffn_dropout <= 0.0f || il < 0 || cur == nullptr || cur->type != GGML_TYPE_F32 ||
        !llm_inj_layer_in_range(c.ffn_layers, il, n_layer)) {
        return cur;
    }

    // keep unit iff sin(K x + phase) > t, with t chosen so that P(keep) = 1 - p for a uniform phase:
    //   P(sin(U) > t) = 1/2 - asin(t)/pi  =>  t = -cos(pi * p)
    const float p     = std::min(c.ffn_dropout, 0.95f);
    const float t     = -std::cos(3.14159265358979f * p);
    const float phase = layer_phase(c.seed, il, 0x66666e64726f70ULL);

    ggml_tensor * u    = ggml_sin(ctx, ggml_scale_bias(ctx, cur, K_FREQ_DROPOUT, phase));
    ggml_tensor * keep = ggml_step(ctx, ggml_scale_bias(ctx, u, 1.0f, -t));
    cur = ggml_scale(ctx, ggml_mul(ctx, cur, keep), 1.0f / (1.0f - p));
    llm_inj_mark_fired(LLM_INJ_SITE_FFN_DROPOUT);
    return cur;
}
