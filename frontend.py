from __future__ import annotations
"""Front-end server for Ethical Parliament (Flask single-file demo).

This file intentionally contains both the Flask routes and the HTML/CSS/JS
via render_template_string so it can run as a single file while we iterate.

Notes
-----
- Black background, neon-accent UI.
- MFQ profile bar chart with worldview toggle + MFQ credit link.
- Scenario entry with guidance (≤5 sentences) and a hard 80-word cap.
- After submit, entry collapses into a confirmation step.
- Circular countdown timer (defaults to 15s for demo; set env var to 300 for 5 min).
- When background job completes, timer is replaced by the faint original prompt and
  the synthesized response.

Safety & Style: matches user's Python house rules where feasible in a single file:
- Uses logging (no print).
- Typed function signatures for public routes.
- Ready for black line-length = 100.
"""
"""Front-end server for Ethical Parliament (Flask single-file demo).

This version omits vectorstore warmup on the frontend, assuming
all semantic / embedding work will be done in scenario builder
or agent-level quote selector logic.
"""

"""Front-end server for Ethical Parliament (Flask single-file with UI)."""


import logging
import os
import time
import uuid
import threading
import json
import pathlib
import subprocess
import re

from datetime import datetime
from flask import Flask, jsonify, render_template_string, request

# Cap parallelism for stability
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("ethical-parliament.frontend")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # limit request size

# Frontend no longer warms vectorstores
STORES = None

def _lazy_warm_vectorstores() -> None:
    logger.info("Skipped vectorstore warmup in frontend")

def ensure_warm() -> None:
    return

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

JOBS_DIR = PROJECT_ROOT / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

def _job_path(job_id: str) -> pathlib.Path:
    return JOBS_DIR / f"{job_id}.json"

def write_status(job_id: str, **fields: Any) -> None:
    try:
        payload = {"job_id": job_id, "last_update": time.time(), **fields}
        _job_path(job_id).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write status for %s: %s", job_id, e)

def _mark_abandoned_jobs_on_boot() -> None:
    for p in JOBS_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("status") == "running":
            d["status"] = "aborted"
            d["reason"] = "server restarted"
            d["last_update"] = time.time()
            p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

_mark_abandoned_jobs_on_boot()

USER_PROFILE_PATH = PROJECT_ROOT / "user_ethics_profile.json"
LATEST_SYNTHESIS_PATH = PROJECT_ROOT / "latest_synthesis.txt"
LATEST_RESULTS_PATH = PROJECT_ROOT / "latest_results.json"
SECRET_TOKEN = os.environ.get("SECRET_TOKEN")
if not SECRET_TOKEN:
    logger.warning("SECRET_TOKEN not set. /start will reject unauthorized calls.")

def sanitize_scenario(text: str) -> str:
    return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text or "").strip()

SIMULATED_DURATION_SEC = int(os.getenv("SIMULATED_DURATION_SEC", "300"))

MFQ_DIMENSIONS = [
    "care_harm",
    "fairness_cheating",
    "loyalty_betrayal",
    "authority_subversion",
    "purity_degradation",
]

NORM_PROFILES = {
    "Liberals (US)": {
        "care_harm": 3.62,
        "fairness_cheating": 3.74,
        "loyalty_betrayal": 2.07,
        "authority_subversion": 2.06,
        "purity_degradation": 1.27,
    },
    "Moderates (US)": {
        "care_harm": 3.31,
        "fairness_cheating": 3.39,
        "loyalty_betrayal": 2.58,
        "authority_subversion": 2.67,
        "purity_degradation": 1.99,
    },
    "Conservatives (US)": {
        "care_harm": 2.98,
        "fairness_cheating": 3.02,
        "loyalty_betrayal": 3.08,
        "authority_subversion": 3.28,
        "purity_degradation": 2.89,
    },
    "Libertarians (US)": {
        "care_harm": 2.80,
        "fairness_cheating": 3.19,
        "loyalty_betrayal": 2.19,
        "authority_subversion": 2.13,
        "purity_degradation": 1.23,
    },
}

DEFAULT_WORLDVIEW = "Moderates (US)"

_JOBS: dict[str, tuple[float, Optional[str], str]] = {}

def _active_profile_from(worldview: str, override: dict[str, float] | None = None) -> dict[str, float]:
    base = dict(NORM_PROFILES.get(worldview, NORM_PROFILES[DEFAULT_WORLDVIEW]))
    if override:
        for k in MFQ_DIMENSIONS:
            v = override.get(k)
            if isinstance(v, (int, float)):
                base[k] = float(v)
    return base

def _start_background_job(job_id: str, prompt: str, profile: dict[str, float]) -> None:
    ready_at = time.time() + SIMULATED_DURATION_SEC
    _JOBS[job_id] = (ready_at, None, prompt)
    write_status(job_id,
                 status="running",
                 progress=0,
                 step="start",
                 eta_seconds=SIMULATED_DURATION_SEC,
                 original_prompt=prompt)
    def worker():
        try:
            subprocess.run(["python", "scenario_builder_general.py", "--scenario", prompt],
                           cwd=str(PROJECT_ROOT), check=True)
            write_status(job_id, status="running", progress=25, step="scenario_built")

            USER_PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
            write_status(job_id, status="running", progress=50, step="profile_saved")

            write_status(job_id, status="running", progress=60, step="synthesis_running")
            subprocess.run(["python", "ethics_synthesis_agent.py"],
                           cwd=str(PROJECT_ROOT), check=True)

            synthesized = ""
            if LATEST_SYNTHESIS_PATH.exists():
                synthesized = LATEST_SYNTHESIS_PATH.read_text(encoding="utf-8").strip()

            if not synthesized and LATEST_RESULTS_PATH.exists():
                try:
                    data = json.loads(LATEST_RESULTS_PATH.read_text(encoding="utf-8"))
                    eq = data.get("ethical_question") or prompt
                    ratings = data.get("agent_ratings") or {}
                    parts = [f"**Ethical Question**\n\n{eq}"]
                    if ratings:
                        parts.append("**Agent Ratings**")
                        for a, r in ratings.items():
                            parts.append(f"- **{a}**: {r}")
                    synthesized = "\n\n".join(parts)
                except Exception:
                    synthesized = ""

            if not synthesized:
                synthesized = "**Pipeline error** – no output"
        except subprocess.CalledProcessError as e:
            synthesized = f"**Pipeline failed (code {e.returncode})**"
            write_status(job_id, status="error", progress=100, step="failed", error=str(e))
        except Exception as e:
            synthesized = f"**Unexpected error**\n{e}"
            write_status(job_id, status="error", progress=100, step="error", error=str(e))
        finally:
            write_status(job_id, status="complete", progress=100, step="done")
            try:
                jf = JOBS_DIR / f"{job_id}.json"
                job_payload = {
                    "status": "complete",
                    "result_markdown": synthesized,
                    "original_prompt": prompt,
                }
                jf.write_text(json.dumps(job_payload, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                logger.warning("Could not write job file %s: %s", job_id, e)
            _JOBS[job_id] = (ready_at, None, "")
            logger.info("Job %s complete.", job_id)
    threading.Thread(target=worker, daemon=True).start()

def _log_mem(tag: str) -> None:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logger.info("[%s] Memory ~ %.1f MB", tag, rss / 1024.0)
    except Exception:
        pass

@app.get("/")
def index() -> str:
    return render_template_string(
        TEMPLATE,
        norm_profiles=NORM_PROFILES,
        mfq_dimensions=MFQ_DIMENSIONS,
        default_worldview=DEFAULT_WORLDVIEW,
        simulated_duration_sec=SIMULATED_DURATION_SEC,
    )

@app.post("/start")
def start():
    payload = request.get_json(force=True, silent=False) or {}
    token = request.headers.get("X-EP-Token") or payload.get("token")
    if not token or token != SECRET_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    scenario = sanitize_scenario(payload.get("scenario") or "")
    worldview = (payload.get("worldview") or DEFAULT_WORLDVIEW).strip()
    override = payload.get("mfq_profile") if isinstance(payload.get("mfq_profile"), dict) else None
    profile = _active_profile_from(worldview, override)

    if not scenario:
        return jsonify({"error": "Scenario is required"}), 400
    if len(scenario.split()) > 80:
        return jsonify({"error": "Please limit to 80 words."}), 400
    # naive sentence count
    sent_count = sum(ch in ".!?" for ch in scenario)
    if sent_count > 5:
        return jsonify({"error": "Please use ≤5 sentences."}), 400

    job_id = uuid.uuid4().hex
    _start_background_job(job_id, scenario, profile)
    logger.info("Started job %s (worldview=%s)", job_id, worldview)
    return jsonify({"job_id": job_id, "eta_seconds": SIMULATED_DURATION_SEC})

@app.get("/status/<job_id>")
def status(job_id: str):
    jp = _job_path(job_id)
    if jp.exists():
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
            return jsonify({
                "status": d.get("status", "pending"),
                "done": d.get("status") == "complete",
                "eta_seconds": d.get("eta_seconds", 0),
                "progress": d.get("progress"),
                "step": d.get("step"),
            })
        except Exception as e:
            logger.warning("Failed read status %s: %s", job_id, e)
    if job_id not in _JOBS:
        return jsonify({"error": "Unknown job"}), 404
    ready_at, result, _ = _JOBS[job_id]
    now = time.time()
    rem = max(0, int(round(ready_at - now))) if result is None else 0
    return jsonify({"status": "complete" if result is not None else "pending", "done": bool(result), "eta_seconds": rem})

@app.get("/result/<job_id>")
def result(job_id: str):
    jf = JOBS_DIR / f"{job_id}.json"
    if jf.exists():
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            if d.get("status") != "complete":
                return jsonify({"status": d.get("status")}), 202
            return jsonify({
                "status": "complete",
                "result_markdown": d.get("result_markdown"),
                "original_prompt": d.get("original_prompt"),
            })
        except Exception as e:
            logger.warning("Failed to read result %s: %s", job_id, e)
    if job_id not in _JOBS:
        return jsonify({"error": "Unknown job"}), 404
    _, resp, prompt = _JOBS[job_id]
    if resp is None:
        return jsonify({"status": "pending"}), 202
    return jsonify({"status": "complete", "result_markdown": resp, "original_prompt": prompt})

@app.get("/jobs")
def list_jobs():
    items = []
    for p in JOBS_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except:
            continue
        items.append({
            "job_id": p.stem,
            "status": d.get("status", "unknown"),
            "last_update": d.get("last_update"),
            "step": d.get("step"),
            "progress": d.get("progress"),
        })
    items.sort(key=lambda x: (x.get("last_update") or 0), reverse=True)
    return jsonify(items)

@app.get("/jobs/<job_id>")
def get_job(job_id: str):
    p = _job_path(job_id)
    if not p.exists():
        return jsonify({"error": "Unknown job"}), 404
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return jsonify(d)
    except Exception as e:
        logger.warning("Failed read job %s: %s", job_id, e)
        return jsonify({"error": "Corrupt job file"}), 500

# ---- Template with embedded CSS / JS ----
TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Ethical Parliament</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      margin: 0;
      padding: 0;
      background-color: #0e0f11;
      color: #e0e0e0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }
    .container {
      max-width: 800px;
      margin: 2rem auto;
      padding: 1.5rem;
      background: #1c1d21;
      border-radius: 8px;
      border: 1px solid #333;
    }
    h1 {
      text-align: center;
      color: #f8f9fa;
    }
    textarea.scenario-input {
      width: 100%;
      padding: 0.75rem;
      background: #2b2c31;
      border: 1px solid #444;
      border-radius: 4px;
      color: #f1f1f1;
      font-size: 1rem;
    }
    .btn {
      margin-top: 1rem;
      background: #5a9efc;
      color: #fff;
      border: none;
      padding: 0.7rem 1.5rem;
      border-radius: 4px;
      cursor: pointer;
    }
    .btn:hover {
      background: #4891e5;
    }
    .response-box {
      margin-top: 2rem;
      background: #232629;
      padding: 1rem;
      border-radius: 6px;
      border: 1px solid #444;
      white-space: pre-wrap;
    }
    .quote-block {
      margin: 1rem 0;
      padding-left: 1rem;
      border-left: 3px solid #5a9efc;
      font-style: italic;
    }
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Ethical Parliament</h1>
    <textarea id="scenario" class="scenario-input" rows="4"
      placeholder="Enter scenario (≤ 5 sentences, ≤ 80 words)"></textarea>
    <br>
    <button id="submit-btn" class="btn">Submit</button>
    <div id="loading" class="response-box hidden">Processing…</div>
    <div id="result" class="response-box hidden"></div>
  </div>
  <script>
    const submitBtn = document.getElementById("submit-btn");
    const scenarioInput = document.getElementById("scenario");
    const loadingDiv = document.getElementById("loading");
    const resultDiv = document.getElementById("result");

    function showLoading() {
      resultDiv.classList.add("hidden");
      loadingDiv.classList.remove("hidden");
      loadingDiv.textContent = "Processing…";
    }
    function showResult(md) {
      loadingDiv.classList.add("hidden");
      resultDiv.classList.remove("hidden");
      resultDiv.innerHTML = md;
    }

    submitBtn.addEventListener("click", async () => {
      const scenario = scenarioInput.value.trim();
      if (!scenario) {
        alert("Please enter a scenario.");
        return;
      }
      showLoading();
      try {
        const resp = await fetch("/start", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-EP-Token": "{{ secret_token }}"
          },
          body: JSON.stringify({ scenario })
        });
        const j = await resp.json();
        if (j.error) {
          showResult("Error: " + j.error);
          return;
        }
        const jobId = j.job_id;
        let done = false;
        while (!done) {
          await new Promise(r => setTimeout(r, 1000));
          const st = await fetch(`/status/${jobId}`);
          const sj = await st.json();
          done = sj.done;
        }
        const rf = await fetch(`/result/${jobId}`);
        const rj = await rf.json();
        showResult(rj.result_markdown || "(no output)");
      } catch (err) {
        showResult("Error: " + err.toString());
      }
    });
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)