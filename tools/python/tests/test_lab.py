import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from llm_injection_lab import COLUMNS, LabClient, LabError, flatten, load_history


def metrics(imp):
    return {"impairment": imp, "divergence": 0.5, "noiseFloor": 0.1, "excessDivergence": 0.4,
            "treated": {"garble": 0.2, "repetition": 0.1, "scriptSwitch": 0, "words": 12},
            "anchor": {"treated": True}, "internal": {"treated": {"entropy": 1.5, "surprisal": 3.0}},
            "clean": {"treated": {"surprisal": 4.0}, "baseline": {"surprisal": 2.0}}}


RUN = {"id": "run-1", "type": "run", "techniqueId": "delirium", "backend": "llamacpp", "model": "m.gguf", "doseMg": 200,
       "intensity": 0.6, "input": {"coTechniqueId": "amnesia", "coDoseMg": 50, "schedule": {"onset": 8, "halfLife": 64}},
       "arms": {"treated": {"seed": 7}}, "metrics": metrics(0.3)}
SWEEP = {"id": "dose-1", "type": "dose-response", "techniqueId": "amnesia", "backend": "ollama", "model": {"name": "x"},
         "input": {}, "rows": [{"doseMg": 0, "trial": 0, "seed": 1, "intensity": 0, "metrics": metrics(0)},
                               {"doseMg": 100, "trial": 1, "seed": 2, "intensity": 0.4, "metrics": metrics(0.2)},
                               {"doseMg": 100, "trial": 2, "seed": 3, "intensity": 0.4, "metrics": None}]}
AGENT = {"id": "agent-1", "type": "agent", "techniqueId": "euphoria", "backend": "llamacpp", "model": "m", "doseMg": 400,
         "intensity": 0.9, "input": {}, "steps": [{"step": 0, "treated": {"seed": 5}, "metrics": metrics(0.5)},
                                                   {"step": 1, "effectiveDoseMg": 200, "intensity": 0.5, "treated": {"seed": 6}, "metrics": metrics(0.1)}]}
LEGACY = {"id": "old", "type": "run", "metrics": {}}


class RecordsTest(unittest.TestCase):
    def test_flatten_matches_csv_columns(self):
        rows = flatten([RUN, SWEEP, AGENT])
        self.assertEqual(len(rows), 1 + 2 + 2)
        for r in rows:
            self.assertEqual(list(r), COLUMNS)
        run = rows[0]
        self.assertEqual((run["dose_mg"], run["co_technique"], run["co_dose_mg"], run["onset_tokens"], run["seed"]), (200, "amnesia", 50, 8, 7))
        self.assertEqual((run["clean_surprisal"], run["clean_surprisal_baseline"], run["anchor"]), (4.0, 2.0, True))
        self.assertEqual(rows[2]["model"], '{"name": "x"}')
        self.assertEqual((rows[3]["dose_mg"], rows[4]["dose_mg"], rows[4]["intensity"]), (400, 200, 0.5))

    def test_load_history_skips_corrupt_and_legacy_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "runs.jsonl"
            p.write_text("\n".join([json.dumps(RUN), "{not json", json.dumps(LEGACY), "", json.dumps(SWEEP)]), encoding="utf-8")
            (Path(d) / "agent-1.json").write_text(json.dumps(AGENT), encoding="utf-8")
            self.assertEqual([r["id"] for r in load_history(p)], ["run-1", "dose-1"])
            self.assertEqual(len(load_history(p, include_legacy=True)), 3)
            self.assertEqual(sorted(r["id"] for r in load_history(d)), ["agent-1", "dose-1", "run-1"])


class FakeLab(BaseHTTPRequestHandler):
    polls = 0

    def log_message(self, *args):
        pass

    def _send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/catalog":
            return self._send(200, {"techniques": [{"id": "delirium"}]})
        if self.path.startswith("/api/history?limit=3"):
            return self._send(200, {"items": [RUN]})
        if self.path == "/api/jobs/job-1":
            FakeLab.polls += 1
            done = FakeLab.polls >= 2
            return self._send(200, {"id": "job-1", "status": "done" if done else "running",
                                    "progress": {"done": FakeLab.polls, "total": 2}, "result": RUN if done else None})
        if self.path == "/api/jobs/job-bad":
            return self._send(200, {"id": "job-bad", "status": "error", "error": "boom"})
        self._send(404, {"error": "Unknown API route"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])) or b"{}")
        if self.path == "/api/run":
            if body.get("doseMg", 0) > 500:
                return self._send(400, {"error": "doseMg must be <= 500"})
            return self._send(202, {"id": "job-1", "status": "queued"})
        self._send(404, {"error": "nope"})


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeLab)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.client = LabClient(f"http://127.0.0.1:{cls.server.server_port}")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_read_endpoints(self):
        self.assertEqual(self.client.techniques()[0]["id"], "delirium")
        self.assertEqual(self.client.history(3)[0]["id"], "run-1")

    def test_run_waits_for_job(self):
        seen = []
        job = self.client.submit("run", techniqueId="delirium", doseMg=200)
        rec = self.client.wait(job["id"], poll=0.01, on_progress=seen.append)
        self.assertEqual(rec["id"], "run-1")
        self.assertGreaterEqual(len(seen), 2)

    def test_errors(self):
        with self.assertRaises(LabError) as ctx:
            self.client.run(doseMg=900)
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("500", str(ctx.exception))
        with self.assertRaises(LabError):
            self.client.wait("job-bad", poll=0.01)
        with self.assertRaises(ValueError):
            self.client.submit("bogus")


if __name__ == "__main__":
    unittest.main()
