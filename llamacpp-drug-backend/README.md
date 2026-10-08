# llama.cpp drug backend artifact

This folder is the lower-level backend design for real inside-the-runtime drugs.

The current app talks to Ollama, which exposes sampler and context knobs but not arbitrary logits, activations, or KV-cache internals. To go deeper, patch an inference runtime such as `llama.cpp` and call a drug hook after the model forward pass and before token sampling.

## Hook points

1. **Logit perturbation**: run `llm_drugs_apply_logits(...)` on the vocabulary logits immediately before sampling.
2. **KV-cache perturbation**: run `llm_drugs_should_drop_kv(...)` or equivalent in the KV-cache update path to drop/corrupt selected memory positions.
3. **Dose config**: map drug + dose to internal coefficients once per request and keep the random seed fixed for reproducibility.

## Drug effects

- **Hallucinogen**: high Gaussian logit noise, tail-token promotion, random top-token suppression.
- **Amnesia**: KV-cache drops, shortened effective context, stronger repeat penalty upstream.
- **Confusion**: logit noise plus top-token suppression, damaging stable reasoning transitions.
- **Creativity Steroid**: tail-token promotion and high entropy.
- **Ego Booster**: entropy compression, making the sampler more decisive.
- **Delusion / Paranoia**: moderate noise plus persistent seed-biased token promotion.

## Integration shape

The exact `llama.cpp` files change over time, but the logic belongs in the sampler chain right before the selected sampler consumes logits. Pseudocode:

```cpp
llama_token_data_array * cur_p = ...;
llm_drugs_config cfg = llm_drugs_from_request(request.drug, request.dose_mg, request.seed);
llm_drugs_apply_logits(cur_p->data, cur_p->size, cfg);
llama_sampler_apply(smpl, cur_p);
```

For KV-cache:

```cpp
if (llm_drugs_should_drop_kv(pos, layer, cfg)) {
    continue; // skip or zero this KV write
}
```

`llm_drug_sampler.h` and `llm_drug_sampler.cpp` are dependency-light C++ modules designed to be copied into a patched inference server.
