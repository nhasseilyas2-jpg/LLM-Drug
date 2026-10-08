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
| What is perturbed | logits, attention, residual stream, FFN, KV memory | sampling options only |
| Techniques | all | placebo (full); hallucinogen, depressant, amnesia, stimulant, delirium, creativity (approximate); dissociative, delusion, paranoia (not available) |
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

## Known limitations

- The drug names are analogies. No claim is made that a technique reproduces the pharmacology or
  phenomenology of any substance.
- Calibration (EC50 and per-technique constants) was done on qwen2.5-0.5b-instruct. Larger models
  are generally more robust; the same dose usually produces a smaller effect. Recalibrate, or
  compare models at matched impairment rather than matched dose.
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
