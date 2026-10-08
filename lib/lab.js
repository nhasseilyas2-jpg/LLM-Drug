import * as metrics from "../src/metrics.js";
import { History } from "./history.js";
import { LlamaServerPool } from "./llamacpp.js";
import { ModelRegistry } from "./models.js";
import { ollamaChat } from "./ollama.js";

const JUDGE_SCHEMA = {
  type: "object",
  properties: {
    A: {
      type: "object",
      properties: {
        coherence: { type: "integer", minimum: 0, maximum: 100 },
        on_task: { type: "integer", minimum: 0, maximum: 100 },
        factual: { type: "integer", minimum: 0, maximum: 100 }
      },
      required: ["coherence", "on_task", "factual"]
    },
    B: {
      type: "object",
      properties: {
        coherence: { type: "integer", minimum: 0, maximum: 100 },
        on_task: { type: "integer", minimum: 0, maximum: 100 },
        factual: { type: "integer", minimum: 0, maximum: 100 }
      },
      required: ["coherence", "on_task", "factual"]
    },
    notes: { type: "string" }
  },
  required: ["A", "B", "notes"]
};

// Shared runtime: model registry, llama-server pool, Ollama client, judge and history.
export class Lab {
  constructor(config) {
    this.config = config;
    this.registry = new ModelRegistry(config);
    this.pool = new LlamaServerPool(config);
    this.history = new History(config);
    this.metrics = metrics;
  }

  ollama(model, request) {
    return ollamaChat(this.config, { model, ...request });
  }

  // Blind judge: the two responses are shown as A/B in a seed-dependent order and the judge is
  // not told which one was treated or which technique was used.
  async judge(input, baseline, treated, signal) {
    const model = input.judgeModelId.slice("ollama:".length);
    const swap = input.seed % 2 === 1;
    const [a, b] = swap ? [treated, baseline] : [baseline, treated];
    const prompt = [
      "Two assistant responses to the same task follow. Score each one independently from 0 to 100 on:",
      "coherence (fluent, grammatical, self-consistent), on_task (addresses the task), factual (correct; use the reference if given).",
      "Reply with JSON only.",
      `Task:\n${input.prompt}`,
      input.expected ? `Reference answer: ${input.expected}` : "",
      `Response A:\n${a}`,
      `Response B:\n${b}`
    ].filter(Boolean).join("\n\n");
    try {
      const r = await this.ollama(model, {
        messages: [{ role: "user", content: prompt }],
        options: { temperature: 0, top_p: 1, top_k: 1, num_ctx: 8192 },
        seed: 90125,
        maxTokens: 400,
        logprobs: false,
        format: JUDGE_SCHEMA,
        signal
      });
      const parsed = JSON.parse(r.content);
      return {
        model,
        blind: true,
        order: swap ? "A=treated,B=baseline" : "A=baseline,B=treated",
        baseline: swap ? parsed.B : parsed.A,
        treated: swap ? parsed.A : parsed.B,
        notes: String(parsed.notes || "").slice(0, 1000)
      };
    } catch (error) {
      return { model, error: error.message };
    }
  }

  shutdown() {
    this.pool.stopAll();
  }
}
