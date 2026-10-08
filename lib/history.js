import { appendFile, mkdir, readFile, rename, stat } from "node:fs/promises";
import path from "node:path";
import { resolveTechniqueId } from "../src/catalog.js";

// Append-only JSONL history. Corrupt lines are skipped, not fatal. Files rotate by size.

export class History {
  constructor(config) {
    this.file = path.join(config.dataDir, "runs.jsonl");
    this.maxBytes = config.historyMaxBytes;
  }

  async append(record) {
    await mkdir(path.dirname(this.file), { recursive: true });
    try {
      const s = await stat(this.file);
      if (s.size > this.maxBytes) {
        await rename(this.file, this.file.replace(/\.jsonl$/, `-${new Date().toISOString().replace(/[:.]/g, "-")}.jsonl`));
      }
    } catch {
      // no file yet
    }
    await appendFile(this.file, `${JSON.stringify(record)}\n`, "utf8");
  }

  async list(limit = 50) {
    let raw = "";
    try {
      raw = await readFile(this.file, "utf8");
    } catch {
      return [];
    }
    const items = [];
    const lines = raw.split(/\r?\n/);
    for (let i = lines.length - 1; i >= 0 && items.length < limit; i -= 1) {
      if (!lines[i].trim()) continue;
      try {
        items.push(normalizeRecord(JSON.parse(lines[i])));
      } catch {
        // skip corrupt line
      }
    }
    return items;
  }

  async get(id) {
    const items = await this.list(100000);
    return items.find((item) => item.id === id) || null;
  }
}

// Records written by the pre-1.0 prototype ("LLM Drugs") are tagged legacy and mapped to the
// current technique ids. Their metrics came from a different (partly circular) formula.
export function normalizeRecord(record) {
  if (record.schema >= 2) return record;
  const legacyId = record.profile?.drugId || record.drugId || null;
  return {
    ...record,
    schema: 1,
    legacy: true,
    techniqueId: legacyId ? resolveTechniqueId(legacyId) || legacyId : null,
    doseMg: record.profile?.doseMg ?? null,
    backend: record.backend || (record.audit?.backend === "ollama-runtime" ? "ollama" : record.audit?.backend || null)
  };
}
