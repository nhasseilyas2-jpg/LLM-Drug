# Methodology

This page describes what a run measures, how the controls work and what the numbers do and do
not mean. If you publish results produced with the lab, please report the items under
[Reporting checklist](#reporting-checklist).

## Design of one trial

Every trial generates the **same messages** (system + optional context + prompt) several times:

| Arm | Engine | Sampling seed | Purpose |
|---|---|---|---|
| **Baseline** | untreated (no `LLM_INJ_*`) | `s` | reference behaviour |
| **Noise floor** | untreated | `s + 100003` | how much two *untreated* samples differ (only when temperature > 0) |
| **Treated** | technique at dose `d`, `LLM_INJ_SEED = s` | `s` | effect of the intervention |

Baseline and treated share the sampling seed, so at temperature 0 any difference is caused by the
intervention alone, and at temperature > 0 the noise floor gives the variability you would see
without any intervention. Untreated arms run in a separate `llama-server` process whose
environment is stripped of every `LLM_INJ_*` variable, so a misconfigured shell cannot leak a
treatment into the control.

**Placebo** is a real arm: it runs the full treated pipeline (own process, same seeds) with
zero intervention. At temperature 0 it must be byte-identical to the baseline; the e2e test suite
checks this on every build.

### Dose-response sweeps

A sweep runs `trials` trials for each dose in the list. Trial `t` uses sampling seed
`s + 7919·t`. Baseline and noise-floor arms are generated once per trial and reused for all doses,
so differences across doses are not confounded by baseline variability.

By default the injection seed stays fixed across trials. Graph-site perturbations depend only on
the injection seed (see the [engine caveats](../llamacpp-injection/README.md#caveats)), so with
temperature 0 all trials would be identical. Enable **reseed injection** to use the trial seed as
the injection seed, which samples a different perturbation per trial.

### Agent loop

The agent mode gives the model an objective and asks for one step at a time, feeding its own
previous steps back as conversation history. Baseline and treated trajectories evolve
independently (the treated model sees its own, possibly derailed, history). Each step is scored
against the baseline step with the same index. This measures how a perturbation compounds over
multiple turns.

**Tolerance** (0–90 %) models a dose that wears off across steps. Step `s` (0-based) receives an
effective dose of `dose · (1 − tolerance)^s`, and so does the co-administered technique. Each step
records its `effectiveDoseMg` and intensity. With tolerance 0, every step gets the full dose.

### Combinations (co-administration)

A second technique and dose can be given with any run, sweep or agent loop. Both translate to
engine settings, which are merged per knob:

| knob kind | rule |
|---|---|
| multiplicative (`ATTN_SCALE`, `LAYER_GAIN`, `LOGIT_TEMP`) | product |
| additive (noise levels, biases) | sum |
| probabilities (`TOP_SUPPRESS`, `KV_FORGET`, dropout, lesion fractions) | `1 − (1 − a)(1 − b)`, capped at 0.95 |
| counts | max |
| `HEAD_GAIN` | 0 (ablation) wins, else product |
| layer ranges | union; if only one side restricts layers, all layers |
| steering | same vector: scales add (euphoria + dysphoria at equal dose cancel); different vectors: primary wins |

In a sweep, the co-dose stays fixed while the primary dose varies, so the curve shows how the
second drug shifts the first one's dose-response (potentiation shows up as a left shift). Values
are clamped to the engine's valid ranges after merging. The merged `treatment.env` is stored with
every run.

### Time course (onset and half-life)

With the patched engine, a run can give the treatment an onset and a half-life, in generated
tokens. The strength of every site is multiplied by
`m(t) = (1 − e^(−ln20·t/onset)) · 0.5^(max(0, t − onset)/half-life)`. The response therefore starts
sober, peaks around `onset`, and wears off. The prompt is processed at `t = 0`, where `m = 0` when
an onset is set, so the prompt itself is never perturbed. The curve is defined in the
[engine README](../llamacpp-injection/README.md#time-course-pharmacokinetics).

### Activation steering (affect)

`euphoria` and `dysphoria` add a dose-scaled "mood" control vector to the residual stream of the
middle layers (20–80 % depth). This is the closest analogue in the lab to a receptor-specific
drug: it pushes the model in one semantic direction instead of adding noise. The vector is the mean
hidden-state difference between paired cheerful and gloomy persona prompts (`steering/mood.json`),
computed per model by `npm run vectors -- --model <name> --vector mood` (about 1 minute for 0.5B
on CPU). It is stored in `data/steering/<model id>/mood.gguf`. Techniques that need a missing
vector are rejected with a hint. Scale is `±2.5·E` on the raw vector. See the engine README for
calibration notes and why the vector is not normalized.

## Metrics

All metrics are computed **from the generated text and logprobs only**. None of them uses the dose,
the technique or the configured parameters, so a score cannot be "built in" by the treatment.
(The pre-1.0 prototype derived part of its score from the dose itself; see
[ISSUES_AND_ROADMAP.md](ISSUES_AND_ROADMAP.md).)

| Metric | Definition | Range |
|---|---|---|
| `divergence` | word-level Levenshtein distance(baseline, treated) / max length | 0–1 |
| `noiseFloor` | same distance between baseline and noise-floor arm | 0–1 |
| `excessDivergence` | `max(0, divergence − noiseFloor)` | 0–1 |
| `garble` | share of whitespace tokens that are implausible (mixed scripts, symbol soup, control/replacement characters, > 30 chars, a character repeated 4+ times) after stripping code blocks, inline code, URLs and markdown | 0–1 |
| `repetition` | `1 − distinct-3` of the word sequence | 0–1 |
| `scriptSwitches` | changes of Unicode script per 100 letters | ≥ 0 |
| `anchor` | 1 if the output contains the expected answer (`a|b` alternatives, case-insensitive) | 0/1 |
| `meanTop1` | mean probability of the most likely token (from logprobs) | 0–1 |
| `entropy` | mean entropy of the renormalized top-5 distribution (nats) | ≥ 0 |
| `surprisal` | mean −log p of the chosen tokens | ≥ 0 |
| `clean.treated.surprisal` | mean −log p of the **treated** text under the **untreated** model (see below) | ≥ 0 |
| `clean.baseline.surprisal` | the same for the baseline text (reference level) | ≥ 0 |

### Clean-model surprisal

With the patched engine, the treated response is fed back to the *untreated* model as a forced
continuation: a GBNF grammar that matches only that exact text, at temperature 0 with logprobs.
The mean −log p of its tokens is how "unlike itself" the treated output is, from the clean
model's point of view. It is dose-blind and needs no reference answer. Unlike the regular
`surprisal`, it also captures logit-site techniques, because the scoring pass is clean. It scores
the single tokenization the grammar produces, so it is an upper bound on the true per-token
surprisal (a lower bound on likelihood). Compare treated against baseline rather than reading
absolute values. It is skipped on Ollama and when the text is empty, and can be turned off with
`cleanScore: false` in the request body. It costs one extra prompt pass per arm.

### Impairment score

A coarse 0–100 summary for the UI and for dose-response plots:

```
impairment = 100 · Σ wᵢ·cᵢ / Σ wᵢ      over the components that are available
  c_div    = excessDivergence                     w = 0.4
  c_garble = clamp(Δgarble / 0.25)                w = 0.2
  c_rep    = clamp(Δrepetition / 0.5)             w = 0.2
  c_anchor = baseline hit and treated miss ? 1 : 0   w = 0.2 (only with an expected answer)
```

`Δ` is treated minus baseline, floored at 0. The weights are a pragmatic choice, not a validated
scale. Prefer the individual metrics (and the judge) for analysis.

### Confidence intervals

Sweeps report the mean and a 95 % percentile bootstrap interval (2000 resamples, deterministic
mulberry32 RNG) per dose. With the default 3 trials these intervals are wide; use ≥ 5 trials and
several prompts for anything you intend to report.

### Logprob caveat

`llama-server` computes logprobs before the sampler chain. They therefore include all graph-site
perturbations (attention, residual, FFN, KV) but **not** logit-site ones (noise, tail boost,
fixation, veto, temperature). For logit-only techniques, compare text metrics instead.

## Blind judge (optional)

An Ollama model scores both responses for coherence, task adherence and factuality (0–100) with
a JSON schema. The judge is not told which response was treated or which technique was used, and
the A/B order flips with the seed parity to cancel position bias. Use a different, preferably
larger, model than the one under test. Judge scores are noisy; treat them as a secondary signal.

## Backends

| | llama.cpp (patched) | Ollama |
|---|---|---|
| What is perturbed | logits, attention, heads, residual stream, FFN, KV memory, steering | sampling options only |
| Techniques | all | placebo (full); hallucinogen, depressant, amnesia, stimulant, delirium, creativity (approximate); dissociative, delusion, paranoia, neurotoxin, stroke, euphoria, dysphoria (not available) |
| Combinations, time course, clean surprisal | yes | no (rejected with 400) |
| Determinism at temperature 0 | yes | yes (same machine and version) |

The Ollama mapping is an approximation: a sampler can only reshape a distribution the model has
already computed. In our runs a confident 30B model (qwen3-coder:30b) was essentially unaffected
by hallucinogen at 400 mg via Ollama (entropy 0.0002 → 0.0004 even at temperature 1.9), while
the same technique via the patched engine clearly degrades a 0.5B model. Use Ollama results as
"sampling stress", not as internal perturbation.

## Reference calibration

`experiments/calibration-llamacpp.json` on qwen2.5-0.5b-instruct-q4_k_m (patched engine, CPU,
3 prompts × 2 trials per dose, seeded). The cells are the mean impairment score (0–100). The last
column is the mean excess divergence at 500 mg (divergence from baseline minus the noise floor).

| technique | 0 mg | 50 mg | 150 mg | 500 mg | excess-div @500 |
|---|---|---|---|---|---|
| placebo | 0.0 | 0.0 | 0.0 | 0.0 | -0.44 |
| hallucinogen | 0.0 | 2.0 | 15.9 | 19.7 | 0.23 |
| depressant | 0.0 | 1.2 | 8.4 | 39.5 | 0.49 |
| amnesia | 0.0 | 14.0 | 31.2 | 38.1 | 0.44 |
| stimulant | 0.0 | 1.5 | 3.7 | 19.0 | 0.30 |
| dissociative | 0.0 | 1.2 | 6.2 | 39.5 | 0.45 |
| delirium | 0.0 | 20.8 | 45.9 | 59.5 | 0.54 |
| delusion | 0.0 | 2.7 | 16.5 | 61.3 | 0.53 |
| paranoia | 0.0 | 1.2 | 10.1 | 50.1 | 0.41 |
| creativity | 0.0 | 0.0 | 4.2 | 8.3 | 0.07 |

Every technique is monotonic in dose, and the placebo stays at zero. Creativity is designed to
be mild and has little effect at temperature 0. Re-run the calibration on your own model before
you compare techniques across models. Potency is not transferable between models.

## Benchmark (robustness curves)

`npm run bench` measures task accuracy, not just divergence. It runs 30 exact-answer tasks
(`bench/exact-answer.json`: arithmetic, facts, spelling and word tasks) with greedy decoding and a
short token budget. An answer counts as correct if, after normalization (case, punctuation,
markdown, `<think>` blocks), it equals an accepted answer, or contains it as a whole word when the
response has at most 6 words. Longer hedged answers do not count. The 0 mg arm is shared by every
technique. Each cell reports accuracy with a Wilson 95 % interval. **D50** is the dose at which
accuracy falls to half of the 0 mg accuracy (linear interpolation between doses, `> max` if never).

Reference run on qwen2.5-0.5b-instruct-q4_k_m (CPU, seed 7), accuracy in %:

| Technique | 0 | 50 | 150 | 300 | 500 | D50 (mg) |
|---|---:|---:|---:|---:|---:|---:|
| Hallucinogen | 60 | 63 | 43 | 40 | 10 | 367 |
| Depressant | 60 | 63 | 43 | 27 | 30 | 270 |
| Amnesic | 60 | 57 | 60 | 47 | 37 | > 500 |
| Stimulant | 60 | 60 | 60 | 53 | 47 | > 500 |
| Dissociative | 60 | 60 | 57 | 7 | 0 | 230 |
| Delirium | 60 | 30 | 7 | 0 | 0 | 50 |
| Delusion | 60 | 60 | 57 | 53 | 40 | > 500 |
| Paranoia | 60 | 60 | 60 | 40 | 20 | 400 |
| Neurotoxin | 60 | 37 | 0 | 0 | 0 | 68 |
| Stroke | 60 | 50 | 40 | 33 | 23 | 367 |
| Euphoria | 60 | 50 | 23 | 0 | 0 | 125 |
| Dysphoria | 60 | 47 | 0 | 0 | 0 | 86 |
| Creativity | 60 | 63 | 60 | 57 | 57 | > 500 |

With 30 tasks, a Wilson interval is roughly ±17 points, so only large differences are meaningful.
Amnesia hardly matters here because the tasks need no earlier context. Its effect shows in the
context-recall prompts of the calibration experiment.

Reduced run on **qwen3-coder:30b** (MoE, ~3B active, Q4_K_M from the Ollama store; first 15 tasks,
CPU), accuracy in %:

| Technique | 0 | 150 | 500 | D50 (mg) | D50 on 0.5B |
|---|---:|---:|---:|---:|---:|
| Hallucinogen | 100 | 100 | 27 | 389 | 367 |
| Depressant | 100 | 100 | 100 | > 500 | 270 |
| Dissociative | 100 | 100 | 87 | > 500 | 230 |
| Delirium | 100 | 93 | 20 | 357 | 50 |
| Neurotoxin | 100 | 80 | 0 | 281 | 68 |
| Amnesic | 100 | 100 | 60 | > 500 | > 500 |

The larger model is much more robust to graph-site techniques: depressant and dissociative (48
layers, so each damped layer matters less) barely register. Logit-heavy techniques (hallucinogen,
delirium) still break it at high dose. EC50 constants tuned on 0.5B therefore overstate potency on
large models. Compare models by D50, not by dose.

## Known limitations

- The drug names are analogies. No claim is made that a technique reproduces the pharmacology or
  phenomenology of any substance.
- Calibration (EC50 and per-technique constants) was done on qwen2.5-0.5b-instruct. Larger models
  are more robust (see the 30B benchmark above: D50 up to 7× higher), so the same dose usually
  produces a smaller effect. Run `npm run bench` on your model and compare models by D50.
- Text metrics are surface measures. `divergence` is high for any change, including harmless
  paraphrase; that is why the noise floor and judge exist.
- `creativity` acts mostly through temperature-like logit changes and has little effect at
  temperature 0 by design.
- `amnesia` only matters if the answer depends on earlier context; use the *context* field with a
  fact the prompt asks about (see `experiments/calibration-llamacpp.json`).

## Reporting checklist

- model file (name, quantization), backend, lab version / git commit, llama.cpp commit
- technique id, doses, trials, prompts (or prompt set), sampling settings, seeds, reseed setting
- the `treatment.env` of each run (stored in history and exports)
- noise floor and baseline metrics, not only the treated arm
- bootstrap intervals and number of trials
