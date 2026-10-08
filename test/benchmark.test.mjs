import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { d50, isCorrect, markdownReport, normalizeAnswer, wilson } from "../lib/benchmark.js";

test("benchmark answers: normalized exact match, short answers may wrap the value", () => {
  assert.equal(normalizeAnswer("**Paris.**"), "paris");
  assert.ok(isCorrect("Paris.", ["paris"]));
  assert.ok(isCorrect("The answer is 43.", ["43"]));
  assert.ok(isCorrect("<think>17+26</think>\n43", ["43"]));
  assert.ok(isCorrect("Eight", ["8", "eight"]));
  assert.ok(!isCorrect("433", ["43"]));
  assert.ok(!isCorrect("", ["43"]));
  assert.ok(!isCorrect("It could be 43 or maybe 44 or 45, hard to say really", ["43"]), "long hedged answers do not count");
  assert.ok(isCorrect("I", ["i"]));
});

test("Wilson interval and D50", () => {
  const w = wilson(15, 30);
  assert.equal(w.p, 0.5);
  assert.ok(w.lo > 0.31 && w.lo < 0.34 && w.hi > 0.66 && w.hi < 0.69);
  assert.deepEqual(wilson(0, 0), { p: null, lo: null, hi: null });
  assert.equal(wilson(30, 30).hi, 1);
  const rows = [{ doseMg: 0, accuracy: 0.8 }, { doseMg: 100, accuracy: 0.6 }, { doseMg: 200, accuracy: 0.2 }];
  assert.equal(d50(rows), 150);
  assert.equal(d50([{ doseMg: 0, accuracy: 0.8 }, { doseMg: 500, accuracy: 0.7 }]), null);
  assert.equal(d50([{ doseMg: 0, accuracy: 0 }]), null);
});

test("benchmark task set is well formed and the report renders", async () => {
  const set = JSON.parse(await readFile(new URL("../bench/exact-answer.json", import.meta.url), "utf8"));
  const ids = new Set();
  for (const t of set.tasks) {
    assert.ok(t.id && t.q && Array.isArray(t.a) && t.a.length > 0, JSON.stringify(t));
    assert.ok(!ids.has(t.id));
    ids.add(t.id);
    for (const a of t.a) assert.ok(isCorrect(a, t.a), `${t.id}: accepted answer '${a}' must score as correct`);
  }
  const md = markdownReport({ taskSet: "x", model: "m", tasks: 2, maxTokens: 8, seed: 1, timestamp: "t", doses: [0, 100],
    techniques: [{ name: "Delirium", d50: null, rows: [{ doseMg: 0, accuracy: 1 }, { doseMg: 100, accuracy: 0.5 }] }] });
  assert.match(md, /\| Delirium \| 100 \| 50 \| > 100 \|/);
});
