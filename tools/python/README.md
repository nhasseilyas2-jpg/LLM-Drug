# llm-injection-lab (Python)

A small stdlib-only client for the lab. Use it to load saved history into Python or pandas, and to run experiments from scripts or notebooks.

```bash
pip install -e tools/python            # add [pandas] for to_dataframe
```

```python
from llm_injection_lab import LabClient, load_history, to_dataframe

# Offline: data/runs.jsonl, an exported record .json, or a directory of them
df = to_dataframe(load_history("data/runs.jsonl"))
df.groupby(["technique", "dose_mg"]).impairment.mean()

# Online: the lab must be running (npm start)
lab = LabClient("http://127.0.0.1:4173")
model = next(m for m in lab.models()["gguf"] if m["name"].startswith("qwen"))
rec = lab.run("dose-response", backend="llamacpp", modelId=model["id"], techniqueId="delirium",
              doses=[0, 100, 300], trials=3, prompt="Explain why the sky is blue.")
```

Rows from `flatten()` and `to_dataframe()` use the same columns as the UI's CSV export, with one row per treated generation. Request bodies use the same fields as `POST /api/run`, `/api/dose-response` and `/api/agent` (see `lib/experiment.js` → `validateInput`).

Run the tests with `python -m unittest discover -s tests` from this directory.
