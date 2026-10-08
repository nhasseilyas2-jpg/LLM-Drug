# llm-injection engine (patched llama.cpp)

A small runtime-perturbation engine compiled into llama.cpp. It perturbs the **computation**
(logits, attention, individual heads, residual stream, feed-forward output, KV memory, activation
steering) while a model generates.
Model weights are never modified. With no `LLM_INJ_*` variable set, every hook is a strict no-op and
the patched build behaves exactly like upstream (verified: byte-identical greedy output).

```
llamacpp-injection/
  src/llm-injection.h          config, site enum, hashing helpers, KV predicate
  src/llm-injection.cpp        env/config-file parsing, reload, audit log, logit site, PK curve
  src/llm-injection-graph.cpp  attention / head / residual / FFN / steering graph sites (ggml ops),
                               hash-noise custom op, GGUF control-vector loader
  patches/llama.cpp.patch      small hooks into llama.cpp (graph, KV cache, sampler, server, CMake)
  patches/LLAMA_CPP_COMMIT     upstream commit the patch is tested against
  test/                        standalone unit tests (no model needed)
```

Build: `scripts/setup-llamacpp-injection.ps1` (Windows) or `scripts/setup-llamacpp-injection.sh`
(Linux/macOS). Both clone llama.cpp, check out the pinned commit, apply the patch, copy `src/` and
build `llama-server` and `llama-cvector-generator` (used to make steering vectors). Add `-Test` /
`--test` to also run the unit tests, and `-Cuda` / `--cuda`, `-Vulkan` / `--vulkan` or `--metal`
for GPU builds. Prebuilt archives are produced by `.github/workflows/release.yml` for `v*` tags.

## Configuration

The configuration comes from `LLM_INJ_*` environment variables. Alternatively, set
`LLM_INJ_CONFIG_FILE` to a file of `KEY=VALUE` lines. That file is re-read (if its mtime or size
changed) at the start of every request, and the engine prints `llm-injection: config gen=N`.
The lab uses this **shared mode** by default: one `llama-server` per model, and it rewrites the
file before each request, so dose changes do not restart the process. Set `LLAMA_SHARED=0` to fall
back to one process per (model, setting). Any value that fails to parse or is out of range is
ignored with a warning (`llm-injection: ignoring invalid NAME='value'`), never silently clamped.

| Variable | Range | Default | Meaning |
|---|---|---|---|
| `LLM_INJ_SEED` | uint64 | 0 | seed for every pseudo-random choice |
| `LLM_INJ_LOG` | 0/1 | 0 | dump the parsed configuration |
| `LLM_INJ_LOGIT_NOISE` | 0–50 | 0 | Gaussian noise on logits, in per-step logit std units |
| `LLM_INJ_LOGIT_TEMP` | 0.05–20 | 1 | extra temperature around the mean logit |
| `LLM_INJ_TOP_SUPPRESS` | 0–1 | 0 | per-step probability of vetoing the top-1 token |
| `LLM_INJ_TAIL_BOOST` / `_TAIL_COUNT` | 0–100 / 0–4096 | 0 / 32 | boost (std units) on `TAIL_COUNT` random tokens per step |
| `LLM_INJ_FIXATION_BIAS` / `_FIXATION_IDS` | −100–100 / ids | 0 | persistent bias (std units) on listed token ids |
| `LLM_INJ_ATTN_SCALE` / `_ATTN_LAYERS` | 0.01–20 / range | 1 / all | multiplier on the QKᵀ softmax scale |
| `LLM_INJ_RESID_NOISE` / `_RESID_LAYERS` | 0–10 / range | 0 / all | residual pseudo-noise, relative RMS |
| `LLM_INJ_LAYER_GAIN` / `_GAIN_LAYERS` | −4–4 / range | 1 / all | gain on each block's residual update |
| `LLM_INJ_FFN_DROPOUT` / `_FFN_LAYERS` | 0–0.95 / range | 0 / all | fraction of FFN output units zeroed |
| `LLM_INJ_KV_FORGET` | 0–1 | 0 | probability a cached position is hidden |
| `LLM_INJ_KV_RECENT` / `_KV_SINK` | ≥0 | 32 / 4 | positions always kept visible |
| `LLM_INJ_HEAD_LESION` / `_HEAD_IDS` | 0–1 / `layer:head,...` | 0 / none | fraction of heads per layer (picked by seed), or explicit heads |
| `LLM_INJ_HEAD_GAIN` / `_HEAD_LAYERS` | −4–4 / range | 0 / all | gain on the selected heads' outputs (0 = ablate) |
| `LLM_INJ_FFN_LESION` | 0–1 | 0 | fraction of FFN output channels silenced for every token (fixed by seed) |
| `LLM_INJ_STEER_FILE` | path | — | GGUF control vector (`direction.<layer>` tensors, as written by `llama-cvector-generator`) |
| `LLM_INJ_STEER_SCALE` / `_STEER_LAYERS` | −10–10 / range | 0 / all | `h' = h + scale · v[layer]` |
| `LLM_INJ_NOISE_MODE` | `sin` / `hash` | `sin` | graph-noise source: `hash` = counter-hash custom op (CPU), independent of activation values |
| `LLM_INJ_PK_ONSET` / `_PK_HALFLIFE` | 0–100000 tokens | 0 / 0 | time course of every site's strength (see below) |
| `LLM_INJ_CONFIG_FILE` | path | — | reloadable `KEY=VALUE` file (shared mode) |

Layer ranges are fractional depth `"0.25:0.75"` or absolute inclusive indices `"L3:L10"`.

`npm run inject-env -- <technique> <dose>` prints the variables for a catalog technique, so the
engine can also be used directly with `llama-server`, `llama-cli` or `llama-completion`.

## Sites and exact math

Notation: `h` = hidden state of one token (vector of size `n_embd`), `rms(h) = sqrt(mean(h²))`,
`σ` = standard deviation of the finite candidate logits at the current step, `E` = technique intensity.

### Logits (`llama-sampler.cpp`, start of the sampler chain)

Applied once per generated token to the full candidate array, before any sampler (top-k, top-p,
temperature, ...) runs. Order:

1. **noise**: `l_i += LOGIT_NOISE · σ · N(0,1)`, keyed by token id (independent of candidate order)
2. **tail boost**: `TAIL_COUNT` uniformly drawn candidates get `+TAIL_BOOST · σ` (redrawn each step)
3. **fixation**: listed ids get `+FIXATION_BIAS · σ` at every step
4. **top-1 veto**: with probability `TOP_SUPPRESS`, the current argmax is set to `min(l) − 4σ`
   (stays finite so grammar fallback still works)
5. **temperature**: `l_i = mean + (l_i − mean) / LOGIT_TEMP`

Randomness: `hash(LLM_INJ_SEED, request seed, step)` (splitmix64), so runs are reproducible and a
different request seed gives a different draw. Masked (`-inf`) candidates stay masked.

### Attention (`build_attn_mha`)

`softmax(s · ATTN_SCALE · QKᵀ + mask)`. `< 1` flattens attention (less selective), `> 1` sharpens it.

### Residual stream (`build_cvec`, end of each block)

- **layer gain**: `h_out = h_in + LAYER_GAIN · (block(h_in) − h_in)`. `0` skips the block,
  `> 1` amplifies it. Never applied to the first block (no input captured) or the last block
  (its output rows are pruned to the requested tokens).
- **noise**: `h' = h + RESID_NOISE · rms(h) · √2 · sin(K·h + φ_layer)`, `K = 4099`.

### Feed-forward (dense and MoE)

Inverted dropout: keep unit iff `sin(K·x + φ_layer) > −cos(π·p)`, then scale by `1/(1−p)`,
`K = 3571`, `p = FFN_DROPOUT`. For a uniform phase this keeps each unit with probability `1 − p`.

With `NOISE_MODE=hash`, the `sin(K·x + φ)` term in residual noise and FFN dropout is replaced by
`unit(hash(seed, layer, channel, bits(x)))`, computed by a ggml custom op. Every channel then gets
an independent draw, and the noise is uncorrelated with the activation's magnitude. Custom ops run
on the CPU backend, so GPU builds pay a transfer cost in this mode.

### Heads (`build_attn`, per-head attention output)

Before the output projection, the output of each selected head is multiplied by `HEAD_GAIN`.
Selection is either explicit (`HEAD_IDS=3:5,10:0`) or random: in each layer of `HEAD_LAYERS`, the
`round(HEAD_LESION · n_head)` heads with the smallest `hash(seed, layer, head)`. The selection is
fixed for the whole run (a "lesion"), unlike dropout.

### FFN lesion

`FFN_LESION` silences the same output channels for every token (`unit(hash(seed, layer, c)) < p`),
unlike `FFN_DROPOUT`, which changes with the activations.

### Steering (`build_cvec`)

`h' = h + STEER_SCALE · m · v[layer]` for layers in `STEER_LAYERS`. `v` is the **raw** vector from
the GGUF file. With `llama-cvector-generator --method mean`, it is the mean activation difference
between paired positive and negative prompts. It is not normalized: a relative rule (`s·‖h‖·v̂`)
broke Qwen at `s = 0.1`, because a few outlier dimensions dominate its residual norm. On
qwen2.5-0.5b with the bundled `mood` vector: ±0.5 gives a subtle shift, ±1 an obvious mood, ±2
mood-dominated text, and ±4 incoherent text. Files are cached by path and mtime. A missing or
malformed file is reported as `ignoring LLM_INJ_STEER_FILE (...)` and the site does not fire.

### Time course (pharmacokinetics)

Every site's strength is multiplied by `m(t) = (1 − e^(−ln20 · t/ONSET)) · 0.5^(max(0, t − ONSET)/HALFLIFE)`,
where `t` is the number of tokens generated so far in the request. The absorption term reaches 95 %
at `t = ONSET`, and the strength then halves every `HALFLIFE` tokens. `0` disables a term. Strengths
are interpolated from 1 (no effect) for multiplicative knobs (attention scale, gains, temperature)
and from 0 for additive ones.

### KV memory (`llama-kv-cache.cpp`, KQ mask)

Cached position `p0` is hidden from a query at position `p1` iff
`p0 ≥ KV_SINK` and `p1 − p0 > KV_RECENT` and `unit(hash(seed, p0)) < KV_FORGET`.
The cache content is untouched; only visibility changes, so the effect is reversible per request.

## Audit trail

Every process prints to stderr:

```
llm-injection: ACTIVE seed=42 logits{noise=0.3 ...} resid_noise{rel=0.06 layers=0.15:0.85}
llm-injection: site fired: logits
llm-injection: site fired: resid_noise
```

or `llm-injection: inactive (no LLM_INJ_* effect configured)`. The lab parses these lines and stores the fired
sites with every run, so a result always shows which interventions actually executed.

## Caveats

- **Default graph noise is a function of the activations.** In the default `sin` mode, residual
  noise and FFN dropout use `sin(K·x + φ)`, which runs on every backend. It behaves like bounded
  zero-mean noise (RMS 1/√2), but it is not independent of the input. Use `NOISE_MODE=hash` for
  input-independent draws. Either way, `φ`/the hash depend only on `LLM_INJ_SEED` and the layer, so
  at temperature 0 with a fixed injection seed, repeated trials are identical. Use `reseedInjection`
  in dose sweeps, or temperature > 0.
- **Logprobs exclude the logit site.** `llama-server` reports logprobs computed before the sampler
  chain. They include graph perturbations (attention, residual, FFN, KV) but not logit-site ones.
- **Backend sampling must stay off.** The logit hook lives in the CPU sampler chain.
- **Grammar sampling** applies the chain twice for rejected tokens; the step counter advances twice.
- **Recurrent / hybrid models** (Mamba, RWKV, ...) have no KV mask to edit, so `KV_FORGET` has no effect.
- **Residual and steering sites need `build_cvec`.** About 31 architectures in this llama.cpp version
  do not call it, so residual noise, layer gain and steering do not fire there. The audit line shows this: the site never
  reports "fired".
- The patch targets the commit in `patches/LLAMA_CPP_COMMIT`. Newer llama.cpp versions may need the
  hunks rebased; the hooks are small and are marked with `// llm-injection:` comments.

## Unit tests

```
cmake -S llamacpp-injection/test -B build/engine-tests
cmake --build build/engine-tests --config Release
build/engine-tests/Release/test-llm-injection     # Windows: .exe; Linux: build/engine-tests/test-llm-injection
```

The tests cover env and config-file parsing, reload, layer ranges, hashing statistics, every logit
operation (no-op when unset, determinism, masking, veto, temperature), the KV predicate, head and
FFN lesion selection, hash noise, the PK curve and steering configuration.
The graph sites are covered end to end by `npm run test:e2e`.
