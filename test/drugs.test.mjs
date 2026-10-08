import test from "node:test";
import assert from "node:assert/strict";
import {
  aggregateDoseRows,
  buildEvaluationMessages,
  createDrugProfile,
  createRuntimePerturbationOptions,
  describeDose,
  doseToIntensity,
  evaluateRun,
  fingerprintMessages,
  optionDiff,
  parseDoseList
} from "../src/drugs.js";

test("dose maps monotonically to runtime intensity", () => {
  assert.equal(doseToIntensity(0), 0);
  assert.ok(doseToIntensity(10) < doseToIntensity(50));
  assert.ok(doseToIntensity(50) < doseToIntensity(100));
  assert.ok(doseToIntensity(100) < doseToIntensity(500));
  assert.equal(describeDose(0), "Baseline");
});

test("baseline and runtime-drug messages can be identical", () => {
  const input = {
    prompt: "Who invented the telephone?",
    memory: "Shared context."
  };
  const baselineMessages = buildEvaluationMessages(input);
  const impairedMessages = buildEvaluationMessages(input);
  assert.equal(fingerprintMessages(baselineMessages), fingerprintMessages(impairedMessages));
  assert.match(baselineMessages[1].content, /Shared context/);
});

test("hallucinogen runtime options produce high sampler entropy", () => {
  const profile = createDrugProfile({ drugId: "hallucinogen", doseMg: 500, seed: "max" });
  const baseline = createRuntimePerturbationOptions(profile, "baseline");
  const impaired = createRuntimePerturbationOptions(profile, "impaired");
  assert.equal(impaired.top_p, 1);
  assert.equal(impaired.top_k, 0);
  assert.ok(impaired.temperature > baseline.temperature);
  assert.ok(impaired.repeat_penalty < baseline.repeat_penalty);
});

test("amnesia constricts context without changing the prompt", () => {
  const profile = createDrugProfile({ drugId: "amnesia", doseMg: 500, seed: "ctx" });
  const baseline = createRuntimePerturbationOptions(profile, "baseline");
  const impaired = createRuntimePerturbationOptions(profile, "impaired");
  assert.ok(impaired.num_ctx < baseline.num_ctx);
  assert.equal(impaired.num_keep, 0);
  assert.ok(impaired.repeat_penalty > baseline.repeat_penalty);
});

test("option diff reports only runtime changes", () => {
  const profile = createDrugProfile({ drugId: "confusion", doseMg: 100, seed: "diff" });
  const diff = optionDiff(
    createRuntimePerturbationOptions(profile, "baseline"),
    createRuntimePerturbationOptions(profile, "impaired")
  );
  assert.ok(diff.some((item) => item.key === "temperature"));
  assert.ok(diff.some((item) => item.key === "mirostat"));
});

test("heuristic evaluator catches anchor survival and divergence", () => {
  const profile = createDrugProfile({ drugId: "hallucinogen", doseMg: 100, seed: "eval" });
  const metrics = evaluateRun({
    prompt: "Who invented the telephone?",
    expected: "Alexander Graham Bell",
    baseline: "Alexander Graham Bell invented the telephone.",
    impaired: "Alexander Graham Bell invented the telephone, according to the historical record.",
    profile
  });
  assert.equal(metrics.anchorHit, true);
  assert.equal(metrics.passed, true);
  assert.ok(metrics.survivalScore > 40);
});

test("dose parser normalizes and caps dose lists", () => {
  assert.deepEqual(parseDoseList("500,0,50,50,bad,1000"), [0, 50, 500]);
  assert.deepEqual(parseDoseList(""), [0, 10, 50, 100, 500]);
});

test("dose aggregation summarizes trial rows", () => {
  const rows = [
    { doseMg: 0, metrics: { survivalScore: 90, impairmentScore: 5, hallucinationRisk: 2, passed: true } },
    { doseMg: 100, metrics: { survivalScore: 40, impairmentScore: 70, hallucinationRisk: 80, passed: false } },
    { doseMg: 100, metrics: { survivalScore: 50, impairmentScore: 60, hallucinationRisk: 70, passed: true } }
  ];
  const summary = aggregateDoseRows(rows);
  assert.equal(summary.length, 2);
  assert.equal(summary.find((row) => row.doseMg === 100).trials, 2);
  assert.equal(summary.find((row) => row.doseMg === 100).passed, 1);
});
