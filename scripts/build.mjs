import { cp, mkdir, rm, stat } from "node:fs/promises";
import { join } from "node:path";

const root = process.cwd();
const src = join(root, "src");
const dist = join(root, "dist");
const requiredFiles = ["index.html", "app.js", "drugs.js", "styles.css"];

await rm(dist, { force: true, recursive: true });
await mkdir(dist, { recursive: true });
await cp(src, dist, { recursive: true });

const checks = await Promise.all(
  requiredFiles.map(async (file) => {
    const info = await stat(join(dist, file));
    return { file, bytes: info.size };
  })
);

for (const check of checks) {
  if (check.bytes === 0) {
    throw new Error(`Build produced an empty file: ${check.file}`);
  }
}

console.log(`Built static LLM Drugs app to ${dist}`);
for (const check of checks) {
  console.log(`- ${check.file}: ${check.bytes} bytes`);
}
