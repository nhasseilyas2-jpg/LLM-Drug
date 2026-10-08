import assert from "node:assert/strict";
import test from "node:test";
import {
  anchorHit,
  bootstrapCI,
  compareArms,
  distinctN,
  garble,
  logprobStats,
  repetition,
  scriptSwitchRate,
  summarizeByDose,
  wordDivergence,
  words
} from "../src/metrics.js";

const CLEAN = "The sky looks blue because air molecules scatter short wavelengths of sunlight more strongly than long ones.";

test("words / distinct-n / repetition", () => {
  assert.deepEqual(words("Don't stop-believing, 42 times!"), ["don't", "stop-believing", "42", "times"]);
  assert.equal(distinctN(["a", "b", "a", "b"], 1), 0.5);
  assert.equal(repetition(words(CLEAN)), 0);
  assert.ok(repetition(words("the cat sat ".repeat(20))) > 0.8);
});

test("garble: clean prose and markdown/code are ~0, corrupted text is high", () => {
  assert.equal(garble(CLEAN), 0);
  const md = "## Steps\n\n- Run `npm test`\n- See https://example.com/a?b=c\n\n```js\nconst x = a?.b ?? {};\n```\n| a | b |\n|---|---|\n| 1 | 2 |";
  assert.ok(garble(md) < 0.1, `markdown garble ${garble(md)}`);
  assert.ok(garble("Theashes1 andsuga45.seven1 ă—.—. kald,No 4 or0ask AsyncStorageof ééééé uren") > 0.3);
  assert.equal(garble("这是一个测试句子。"), 0);
});

test("script switching detects mixed scripts", () => {
  assert.equal(scriptSwitchRate(CLEAN), 0);
  assert.ok(scriptSwitchRate("hello мир hello 世界 hello") > 5);
});

test("word divergence is a normalized edit distance", () => {
  assert.equal(wordDivergence("a b c", "a b c"), 0);
  assert.equal(wordDivergence("a b c d", "a x c d"), 0.25);
  assert.equal(wordDivergence("", "a"), 1);
  assert.equal(wordDivergence("", ""), 0);
});

test("anchor hit supports alternatives and returns null without an anchor", () => {
  assert.equal(anchorHit("The answer is Six.", "6|six"), 1);
  assert.equal(anchorHit("The answer is 7.", "6|six"), 0);
  assert.equal(anchorHit("whatever", ""), null);
});

test("logprob stats: confident steps have low entropy", () => {
  const confident = [{ logprob: -0.01, top_logprobs: [{ logprob: -0.01 }, { logprob: -6 }, { logprob: -7 }] }];
  const unsure = [{ logprob: -1.2, top_logprobs: [{ logprob: -1.1 }, { logprob: -1.2 }, { logprob: -1.3 }] }];
  const a = logprobStats(confident);
  const b = logprobStats(unsure);
  assert.ok(a.entropy < 0.1 && b.entropy > 1);
  assert.ok(a.meanTop1 > 0.95 && b.surprisal > a.surprisal);
  assert.equal(logprobStats([]), null);
  assert.equal(logprobStats(null), null);
});

test("compareArms is not circular: identical arms score zero impairment", () => {
  const m = compareArms({ baseline: CLEAN, treated: CLEAN, noise: CLEAN, expected: "blue" });
  assert.equal(m.impairment, 0);
  assert.equal(m.divergence, 0);
  assert.deepEqual(m.anchor, { baseline: 1, treated: 1 });
});

test("compareArms: noise floor is subtracted and anchor loss counts", () => {
  const treated = "Theashes1 andsuga45.seven1 ă—.—. kald,No 4 or0ask";
  const m = compareArms({ baseline: CLEAN, treated, noise: CLEAN.replace("strongly", "much"), expected: "blue" });
  assert.ok(m.noiseFloor > 0 && m.excessDivergence < m.divergence);
  assert.deepEqual(m.anchor, { baseline: 1, treated: 0 });
  assert.ok(m.impairment > 60, `impairment ${m.impairment}`);
});

test("bootstrap CI is deterministic and brackets the mean", () => {
  const v = [0.1, 0.4, 0.35, 0.2, 0.5, 0.3];
  const a = bootstrapCI(v, { seed: 1 });
  assert.deepEqual(a, bootstrapCI(v, { seed: 1 }));
  assert.ok(a.lo <= a.mean && a.mean <= a.hi);
  assert.deepEqual(bootstrapCI([]), { n: 0, mean: null, lo: null, hi: null });
  assert.equal(bootstrapCI([2]).lo, null);
});

test("summarizeByDose groups rows and sorts by dose", () => {
  const m = (x) => compareArms({ baseline: CLEAN, treated: x });
  const rows = [
    { doseMg: 500, metrics: m("zz qq") },
    { doseMg: 0, metrics: m(CLEAN) },
    { doseMg: 500, metrics: m("yy") }
  ];
  const s = summarizeByDose(rows);
  assert.deepEqual(s.map((x) => [x.doseMg, x.n]), [[0, 1], [500, 2]]);
  assert.equal(s[0].impairment.mean, 0);
  assert.ok(s[1].divergence.mean > 0.9);
});
