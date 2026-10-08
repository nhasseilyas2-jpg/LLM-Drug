import { fetchJson } from "./config.js";

export async function ollamaListModels(config) {
  const data = await fetchJson(`${config.ollamaUrl}/api/tags`, { timeoutMs: 5000 });
  return (data?.models || []).map((m) => ({
    name: m.name,
    size: m.size,
    family: m.details?.family || null,
    parameterSize: m.details?.parameter_size || null,
    quantization: m.details?.quantization_level || null
  }));
}

// One chat completion. options = Ollama sampling options; returns { content, logprobs, usage }.
export async function ollamaChat(config, { model, messages, options, seed, maxTokens, logprobs = true, format, signal }) {
  const body = {
    model,
    messages,
    stream: false,
    think: false,
    options: { ...options, seed, num_predict: maxTokens }
  };
  if (logprobs) {
    body.logprobs = true;
    body.top_logprobs = 5;
  }
  if (format) body.format = format;
  let data;
  try {
    data = await fetchJson(`${config.ollamaUrl}/api/chat`, { method: "POST", body, timeoutMs: config.requestTimeoutMs, signal });
  } catch (error) {
    // Older Ollama versions / models without a thinking switch reject "think"; retry without it.
    if (!/think/i.test(error.message)) throw error;
    delete body.think;
    data = await fetchJson(`${config.ollamaUrl}/api/chat`, { method: "POST", body, timeoutMs: config.requestTimeoutMs, signal });
  }
  return {
    content: data?.message?.content ?? "",
    logprobs: Array.isArray(data?.logprobs) ? data.logprobs : null,
    usage: { promptTokens: data?.prompt_eval_count ?? null, completionTokens: data?.eval_count ?? null },
    finishReason: data?.done_reason ?? null
  };
}
