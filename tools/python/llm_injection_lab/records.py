"""Read lab history (data/runs.jsonl or exported .json files) and flatten it to rows.

The columns match the CSV export in the web UI (src/app.js ``recordToCsv``), one row
per treated generation: a single run, each dose x trial of a sweep, or each agent step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

COLUMNS = [
    "id", "type", "technique", "backend", "model", "dose_mg", "co_technique", "co_dose_mg",
    "onset_tokens", "half_life_tokens", "trial", "seed", "intensity", "impairment", "divergence",
    "noise_floor", "excess_divergence", "garble", "repetition", "script_switch", "anchor",
    "entropy", "surprisal", "clean_surprisal", "clean_surprisal_baseline", "words",
]


def _iter_json(path: Path) -> Iterator[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
        yield from (data if isinstance(data, list) else [data])
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue  # the server skips corrupt lines too


def load_history(path: str | Path, include_legacy: bool = False) -> list[dict]:
    """Load records from a runs*.jsonl file, an exported record .json, or a directory of either.

    Records written before schema 2 (no ``input`` field) are skipped unless ``include_legacy``.
    """
    p = Path(path)
    files = sorted([*p.glob("*.jsonl"), *p.glob("*.json")]) if p.is_dir() else [p]
    out = []
    for f in files:
        for rec in _iter_json(f):
            if isinstance(rec, dict) and (include_legacy or "input" in rec):
                out.append(rec)
    return out


def _get(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _row(r: dict, dose: Any, trial: Any, seed: Any, intensity: Any, m: dict) -> dict:
    inp = r.get("input") or {}
    sched = inp.get("schedule") or {}
    model = r.get("model")
    anchor = m.get("anchor")
    return {
        "id": r.get("id"),
        "type": r.get("type"),
        "technique": r.get("techniqueId"),
        "backend": r.get("backend"),
        "model": model if isinstance(model, str) else json.dumps(model),
        "dose_mg": dose,
        "co_technique": inp.get("coTechniqueId"),
        "co_dose_mg": inp.get("coDoseMg"),
        "onset_tokens": sched.get("onset"),
        "half_life_tokens": sched.get("halfLife"),
        "trial": trial,
        "seed": seed,
        "intensity": intensity,
        "impairment": m.get("impairment"),
        "divergence": m.get("divergence"),
        "noise_floor": m.get("noiseFloor"),
        "excess_divergence": m.get("excessDivergence"),
        "garble": _get(m, "treated", "garble"),
        "repetition": _get(m, "treated", "repetition"),
        "script_switch": _get(m, "treated", "scriptSwitch"),
        "anchor": anchor.get("treated") if isinstance(anchor, dict) else None,
        "entropy": _get(m, "internal", "treated", "entropy"),
        "surprisal": _get(m, "internal", "treated", "surprisal"),
        "clean_surprisal": _get(m, "clean", "treated", "surprisal"),
        "clean_surprisal_baseline": _get(m, "clean", "baseline", "surprisal"),
        "words": _get(m, "treated", "words"),
    }


def flatten(records: Iterable[dict]) -> list[dict]:
    """One dict per treated generation, keyed by ``COLUMNS``."""
    rows = []
    for r in records:
        kind = r.get("type")
        if kind == "run" and r.get("metrics"):
            rows.append(_row(r, r.get("doseMg"), 0, _get(r, "arms", "treated", "seed"), r.get("intensity"), r["metrics"]))
        elif kind == "dose-response":
            for row in r.get("rows") or []:
                if row.get("metrics"):
                    rows.append(_row(r, row.get("doseMg"), row.get("trial"), row.get("seed"), row.get("intensity"), row["metrics"]))
        elif kind == "agent":
            for s in r.get("steps") or []:
                if s.get("metrics"):
                    rows.append(_row(r, s.get("effectiveDoseMg", r.get("doseMg")), s.get("step"), _get(s, "treated", "seed"),
                                     s.get("intensity", r.get("intensity")), s["metrics"]))
    return rows


def to_dataframe(records: Iterable[dict]):
    """``flatten`` as a pandas DataFrame (requires pandas)."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("to_dataframe needs pandas: pip install pandas") from exc
    return pd.DataFrame(flatten(records), columns=COLUMNS)
