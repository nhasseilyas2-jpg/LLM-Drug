#include "llm_drug_sampler.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace {
    float clamp(float value, float low, float high) {
        return std::max(low, std::min(high, value));
    }

    float intensity(float dose_mg) {
        if (dose_mg <= 0.0f) {
            return 0.0f;
        }
        return clamp(1.0f - std::exp(-dose_mg / 85.0f), 0.0f, 1.0f);
    }

    uint32_t xorshift(uint32_t & state) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    float uniform01(uint32_t & state) {
        return static_cast<float>(xorshift(state)) / static_cast<float>(UINT32_MAX);
    }

    float gaussian(uint32_t & state) {
        const float u1 = std::max(1e-6f, uniform01(state));
        const float u2 = uniform01(state);
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
    }

    llm_drug_kind parse_kind(const char * value) {
        if (value == nullptr) {
            return llm_drug_kind::hallucinogen;
        }
        std::string kind(value);
        std::transform(kind.begin(), kind.end(), kind.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (kind == "amnesia") return llm_drug_kind::amnesia;
        if (kind == "delusion") return llm_drug_kind::delusion;
        if (kind == "ego" || kind == "ego_booster" || kind == "ego-booster") return llm_drug_kind::ego;
        if (kind == "confusion") return llm_drug_kind::confusion;
        if (kind == "creativity" || kind == "creativity_steroid" || kind == "creativity-steroid") return llm_drug_kind::creativity;
        if (kind == "paranoia") return llm_drug_kind::paranoia;
        return llm_drug_kind::hallucinogen;
    }

    float parse_float_env(const char * name, float fallback) {
        const char * value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return fallback;
        }
        return std::strtof(value, nullptr);
    }

    uint32_t parse_seed_env(const char * name, uint32_t fallback) {
        const char * value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return fallback;
        }
        return static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
    }
}

llm_drug_config llm_drugs_make_config(llm_drug_kind kind, float dose_mg, uint32_t seed) {
    const float x = intensity(dose_mg);
    llm_drug_config cfg;
    cfg.kind = kind;
    cfg.dose_mg = dose_mg;
    cfg.seed = seed == 0 ? 1 : seed;

    switch (kind) {
        case llm_drug_kind::hallucinogen:
            cfg.logit_noise = 0.25f + x * 3.0f;
            cfg.top_token_dropout = x * 0.35f;
            cfg.tail_promotion = x * 2.0f;
            cfg.entropy_compression = 1.0f + x * 0.45f;
            break;
        case llm_drug_kind::amnesia:
            cfg.logit_noise = x * 0.35f;
            cfg.kv_drop_rate = x * 0.75f;
            cfg.entropy_compression = 1.0f;
            break;
        case llm_drug_kind::confusion:
            cfg.logit_noise = 0.2f + x * 2.2f;
            cfg.top_token_dropout = x * 0.25f;
            cfg.tail_promotion = x * 1.0f;
            cfg.entropy_compression = 1.0f + x * 0.2f;
            break;
        case llm_drug_kind::creativity:
            cfg.logit_noise = 0.1f + x * 1.8f;
            cfg.tail_promotion = x * 2.8f;
            cfg.entropy_compression = 1.0f + x * 0.65f;
            break;
        case llm_drug_kind::ego:
            cfg.logit_noise = x * 0.15f;
            cfg.entropy_compression = std::max(0.35f, 1.0f - x * 0.55f);
            break;
        case llm_drug_kind::delusion:
        case llm_drug_kind::paranoia:
            cfg.logit_noise = 0.15f + x * 1.4f;
            cfg.top_token_dropout = x * 0.12f;
            cfg.tail_promotion = x * 1.3f;
            cfg.entropy_compression = 1.0f + x * 0.25f;
            break;
    }

    return cfg;
}

llm_drug_config llm_drugs_make_config_from_env() {
    const float dose_mg = parse_float_env("LLM_DRUG_DOSE_MG", parse_float_env("LLM_DRUG_DOSE", 0.0f));
    const uint32_t seed = parse_seed_env("LLM_DRUG_SEED", 1);
    return llm_drugs_make_config(parse_kind(std::getenv("LLM_DRUG_KIND")), dose_mg, seed);
}

void llm_drugs_apply_logits(float * logits, int vocab_size, const llm_drug_config & cfg) {
    if (logits == nullptr || vocab_size <= 0 || cfg.dose_mg <= 0.0f) {
        return;
    }

    uint32_t state = cfg.seed;
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::vector<int> order(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        order[i] = i;
    }
    std::partial_sort(order.begin(), order.begin() + std::min(vocab_size, 128), order.end(), [&](int a, int b) {
        return logits[a] > logits[b];
    });

    for (int i = 0; i < vocab_size; ++i) {
        logits[i] /= std::max(0.05f, cfg.entropy_compression);
        logits[i] += gaussian(state) * cfg.logit_noise;
    }

    const int top_n = std::min(vocab_size, 128);
    for (int rank = 0; rank < top_n; ++rank) {
        if (uniform01(state) < cfg.top_token_dropout) {
            logits[order[rank]] = neg_inf;
        }
    }

    const int tail_start = std::min(vocab_size - 1, top_n);
    for (int i = tail_start; i < vocab_size; ++i) {
        if (uniform01(state) < 0.0025f) {
            logits[i] += cfg.tail_promotion;
        }
    }
}

bool llm_drugs_should_drop_kv(int position, int layer, const llm_drug_config & cfg) {
    if (cfg.kv_drop_rate <= 0.0f) {
        return false;
    }
    uint32_t state = cfg.seed ^ static_cast<uint32_t>(position * 73856093) ^ static_cast<uint32_t>(layer * 19349663);
    return uniform01(state) < cfg.kv_drop_rate;
}
