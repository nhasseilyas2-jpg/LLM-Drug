#pragma once

#include <cstdint>
#include <vector>

enum class llm_drug_kind {
    hallucinogen,
    amnesia,
    delusion,
    ego,
    confusion,
    creativity,
    paranoia,
};

struct llm_drug_config {
    llm_drug_kind kind = llm_drug_kind::hallucinogen;
    float dose_mg = 0.0f;
    uint32_t seed = 1;
    float logit_noise = 0.0f;
    float top_token_dropout = 0.0f;
    float tail_promotion = 0.0f;
    float entropy_compression = 1.0f;
    float kv_drop_rate = 0.0f;
};

llm_drug_config llm_drugs_make_config(llm_drug_kind kind, float dose_mg, uint32_t seed);
llm_drug_config llm_drugs_make_config_from_env();
void llm_drugs_apply_logits(float * logits, int vocab_size, const llm_drug_config & cfg);
bool llm_drugs_should_drop_kv(int position, int layer, const llm_drug_config & cfg);
