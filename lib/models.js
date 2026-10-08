import { createHash } from "node:crypto";
import { open, readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { ollamaListModels } from "./ollama.js";
import { listVectors } from "./steering.js";

// Model registry. Clients only ever see and send opaque ids; paths stay on the server.

const GGUF_MAGIC = "GGUF";

async function isGguf(file) {
  let handle;
  try {
    handle = await open(file, "r");
    const buf = Buffer.alloc(4);
    await handle.read(buf, 0, 4, 0);
    return buf.toString("latin1") === GGUF_MAGIC;
  } catch {
    return false;
  } finally {
    await handle?.close();
  }
}

export const idFor = (file) => `gguf:${createHash("sha256").update(path.resolve(file).toLowerCase()).digest("hex").slice(0, 16)}`;

async function scanDir(dir, depth = 2) {
  const out = [];
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory() && depth > 0) out.push(...(await scanDir(full, depth - 1)));
    else if (entry.isFile() && /\.gguf$/i.test(entry.name) && !/mmproj/i.test(entry.name)) out.push(full);
  }
  return out;
}

// Ollama stores weights as content-addressed blobs; the manifest names the model layer.
async function ollamaBlobs(modelsDir) {
  const root = path.join(modelsDir, "manifests");
  const files = [];
  const walk = async (dir, parts) => {
    let entries;
    try { entries = await readdir(dir, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) await walk(full, [...parts, e.name]);
      else files.push({ full, parts: [...parts, e.name] });
    }
  };
  await walk(root, []);
  const out = [];
  for (const { full, parts } of files) {
    try {
      const manifest = JSON.parse(await readFile(full, "utf8"));
      const layer = (manifest.layers || []).find((l) => l.mediaType === "application/vnd.ollama.image.model");
      if (!layer?.digest) continue;
      const blob = path.join(modelsDir, "blobs", layer.digest.replace(":", "-"));
      if (!(await isGguf(blob))) continue;
      const [host, namespace, name, tag] = parts.slice(-4);
      const label = `${namespace === "library" ? "" : `${namespace}/`}${name}:${tag}`;
      out.push({ file: blob, label, host });
    } catch {
      // unreadable manifest: skip
    }
  }
  return out;
}

export class ModelRegistry {
  constructor(config) {
    this.config = config;
    this.gguf = new Map();
  }

  async refresh() {
    const found = new Map();
    for (const dir of this.config.modelsDirs) {
      for (const file of await scanDir(dir)) {
        if (!(await isGguf(file))) continue;
        const s = await stat(file);
        found.set(idFor(file), { id: idFor(file), name: path.basename(file), source: "models", size: s.size, file });
      }
    }
    for (const blob of await ollamaBlobs(this.config.ollamaModelsDir)) {
      const s = await stat(blob.file);
      found.set(idFor(blob.file), { id: idFor(blob.file), name: `${blob.label} (Ollama blob)`, source: "ollama-blob", size: s.size, file: blob.file });
    }
    this.gguf = found;
    let ollama = [];
    let ollamaError = null;
    try {
      ollama = (await ollamaListModels(this.config)).map((m) => ({ id: `ollama:${m.name}`, ...m }));
    } catch (error) {
      ollamaError = error.message;
    }
    return {
      gguf: await Promise.all([...found.values()].map(async ({ file, ...rest }) => ({ ...rest, steering: await listVectors(this.config, rest.id) }))),
      ollama,
      ollamaError
    };
  }

  resolveGguf(id) {
    const entry = this.gguf.get(id);
    if (!entry) throw new Error(`Unknown GGUF model id '${id}'. Refresh the model list.`);
    return entry;
  }
}
