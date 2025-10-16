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

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple

# Hard-cap parallelism to keep memory stable
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

import json
import pathlib
import subprocess
from datetime import datetime
import re

from flask import Flask, jsonify, render_template_string, request

# -----------------------------------------------------------------------------
# App setup & logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("ethical-parliament.frontend")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # 64 KB

# -----------------------------------------------------------------------------
# No vectorstore logic in frontend
# -----------------------------------------------------------------------------
STORES = None

def _lazy_warm_vectorstores() -> None:
    """Stub: vectorstore warmup skipped on frontend."""
    logger.info("Frontend: vectorstore warmup skipped")

def ensure_warm() -> None:
    """No-op in new architecture."""
    return

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

JOBS_DIR = PROJECT_ROOT / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

def _job_path(job_id: str) -> pathlib.Path:
    return JOBS_DIR / f"{job_id}.json"

def write_status(job_id: str, **fields: Any) -> None:
    """Persist minimal status record for this job id."""
    try:
        payload = {"job_id": job_id, "last_update": time.time(), **fields}
        _job_path(job_id).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write status for %s: %s", job_id, e)

def _mark_abandoned_jobs_on_boot() -> None:
    """Flip lingering running jobs to aborted if server restarted."""
    try:
        for p in JOBS_DIR.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("status") == "running":
                data["status"] = "aborted"
                data["reason"] = "server restarted"
                data["last_update"] = time.time()
                p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to mark abandoned jobs: %s", e)

_mark_abandoned_jobs_on_boot()

USER_PROFILE_PATH = PROJECT_ROOT / "user_ethics_profile.json"
LATEST_SYNTHESIS_PATH = PROJECT_ROOT / "latest_synthesis.txt"
LATEST_RESULTS_PATH = PROJECT_ROOT / "latest_results.json"
SECRET_TOKEN = os.environ.get("SECRET_TOKEN")
if not SECRET_TOKEN:
    logger.warning("SECRET_TOKEN not set; /start will reject requests without a valid token.")

# -----------------------------------------------------------------------------
# Scenario sanitizer
# -----------------------------------------------------------------------------
def sanitize_scenario(text: str) -> str:
    """Strip control chars (0–31, DEL) and trim."""
    return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text or "").strip()

SIMULATED_DURATION_SEC: int = int(os.getenv("SIMULATED_DURATION_SEC", "300"))

MFQ_DIMENSIONS = [
    "care_harm",
    "fairness_cheating",
    "loyalty_betrayal",
    "authority_subversion",
    "purity_degradation",
]

NORM_PROFILES: Dict[str, Dict[str, float]] = {
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

_JOBS: Dict[str, Tuple[float, Optional[str], str]] = {}

def _active_profile_from(worldview: str, override: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    base = dict(
        NORM_PROFILES.get(
            worldview,
            NORM_PROFILES.get(DEFAULT_WORLDVIEW, next(iter(NORM_PROFILES.values())))
        )
    )
    if override:
        for k in MFQ_DIMENSIONS:
            v = override.get(k)
            if isinstance(v, (int, float)):
                base[k] = float(v)
    return base

def _start_background_job(job_id: str, prompt: str, profile: Dict[str, float]) -> None:
    ready_at = time.time() + SIMULATED_DURATION_SEC
    _JOBS[job_id] = (ready_at, None, prompt)
    write_status(
        job_id,
        status="running",
        progress=0,
        step="start",
        eta_seconds=SIMULATED_DURATION_SEC,
        original_prompt=prompt,
    )

    def _worker() -> None:
        try:
            # 1) Build scenario
            subprocess.run(
                ["python", "scenario_builder_general.py", "--scenario", prompt],
                cwd=str(PROJECT_ROOT),
                check=True,
            )
            write_status(job_id, status="running", progress=25, step="scenario_built")

            # 2) Save user profile for MFQ
            USER_PROFILE_PATH.write_text(
                json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            write_status(job_id, status="running", progress=50, step="profile_saved")

            # 3) Run synthesis pipeline
            write_status(job_id, status="running", progress=60, step="synthesis_running")
            subprocess.run(
                ["python", "ethics_synthesis_agent.py"], cwd=str(PROJECT_ROOT), check=True
            )

            # 4) Read synthesis
            synthesized = ""
            if LATEST_SYNTHESIS_PATH.exists():
                synthesized = LATEST_SYNTHESIS_PATH.read_text(encoding="utf-8").strip()

            # fallback: build minimal markdown from latest_results
            if not synthesized and LATEST_RESULTS_PATH.exists():
                try:
                    data = json.loads(LATEST_RESULTS_PATH.read_text(encoding="utf-8"))
                    eq = data.get("ethical_question") or prompt
                    ratings = data.get("agent_ratings") or {}
                    parts = [f"**Ethical Question**\n\n{eq}"]
                    if ratings:
                        parts.append("**Agent Ratings**")
                        for agent, rating in ratings.items():
                            parts.append(f"- **{agent}**: {rating}")
                    synthesized = "\n\n".join(parts).strip()
                except Exception:
                    synthesized = ""

            if not synthesized:
                synthesized = (
                    "**Pipeline error**\n\n"
                    "The backend did not produce a synthesis. Please inspect logs."
                )
        except subprocess.CalledProcessError as e:
            synthesized = f"**Pipeline failed (exit {e.returncode})**"
            write_status(job_id, status="error", progress=100, step="failed", error=str(e))
        except Exception as e:
            synthesized = f"**Unexpected error**\n\n{e}"
            write_status(job_id, status="error", progress=100, step="error", error=str(e))
        finally:
            write_status(job_id, status="complete", progress=100, step="done")
            try:
                job_file = JOBS_DIR / f"{job_id}.json"
                job_payload = {
                    "status": "complete",
                    "result_markdown": synthesized,
                    "original_prompt": prompt,
                }
                job_file.write_text(json.dumps(job_payload, ensure_ascii=False), encoding="utf-8")
            except Exception as write_err:
                logger.warning("Failed to write job file %s: %s", job_id, write_err)

            _JOBS[job_id] = (ready_at, None, "")
            logger.info("Job %s completed.", job_id)

    threading.Thread(target=_worker, daemon=True).start()

def _log_mem(tag: str) -> None:
    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logger.info("[%s] RSS ~ %.1f MB", tag, rss_kb / 1024.0)
    except Exception:
        pass

@app.get("/")
def index() -> str:
    """Serve the main UI."""
    # No ensure_warm() call here
    return render_template_string(
        TEMPLATE,
        norm_profiles=NORM_PROFILES,
        mfq_dimensions=MFQ_DIMENSIONS,
        default_worldview=DEFAULT_WORLDVIEW,
        simulated_duration_sec=SIMULATED_DURATION_SEC,
    )

def _count_words(text: str) -> int:
    return len([w for w in text.strip().split() if w])

def _count_sentences_naive(text: str) -> int:
    count = 0
    for ch in text:
        if ch in ".!?":
            count += 1
    return max(1, count) if text.strip() else 0

@app.post("/start")
def start() -> Any:
    payload: Dict[str, Any] = request.get_json(force=True, silent=False) or {}
    token = request.headers.get("X-EP-Token") or (payload.get("token") if isinstance(payload.get("token"), str) else None)
    if not token or token != SECRET_TOKEN:
        return jsonify({"error": "Unauthorized."}), 401

    scenario = sanitize_scenario(payload.get("scenario") or "")
    worldview = (payload.get("worldview") or DEFAULT_WORLDVIEW).strip()
    override_profile = payload.get("mfq_profile") if isinstance(payload.get("mfq_profile"), dict) else None
    active_profile = _active_profile_from(worldview, override_profile)

    if not scenario:
        return jsonify({"error": "Scenario text is required."}), 400
    if _count_words(scenario) > 80:
        return jsonify({"error": "Please keep to 80 words or fewer."}), 400
    if _count_sentences_naive(scenario) > 5:
        return jsonify({"error": "Please use five sentences or fewer."}), 400

    job_id = uuid.uuid4().hex
    _start_background_job(job_id, scenario, active_profile)
    logger.info("Started job %s (worldview=%s, %d words)", job_id, worldview, _count_words(scenario))

    return jsonify({"job_id": job_id, "eta_seconds": SIMULATED_DURATION_SEC})

@app.get("/status/<job_id>")
def status(job_id: str) -> Any:
    job_file = _job_path(job_id)
    if job_file.exists():
        try:
            data = json.loads(job_file.read_text(encoding="utf-8"))
            resp = {
                "status": data.get("status", "pending"),
                "done": data.get("status") == "complete",
                "eta_seconds": data.get("eta_seconds", 0),
                "progress": data.get("progress"),
                "step": data.get("step"),
            }
            return jsonify(resp)
        except Exception as e:
            logger.warning("Failed to read status file %s: %s", job_id, e)
    if job_id not in _JOBS:
        return jsonify({"error": "Unknown job id."}), 404
    ready_at, result, _ = _JOBS[job_id]
    now = time.time()
    remaining = max(0, int(round(ready_at - now))) if result is None else 0
    return jsonify({"status": "complete" if result is not None else "pending", "done": bool(result is not None), "eta_seconds": remaining})

@app.get("/result/<job_id>")
def result(job_id: str) -> Any:
    job_file = JOBS_DIR / f"{job_id}.json"
    if job_file.exists():
        try:
            data = json.loads(job_file.read_text(encoding="utf-8"))
            if data.get("status") != "complete":
                return jsonify({"status": data.get("status")}), 202
            return jsonify({"status": "complete", "result_markdown": data.get("result_markdown", ""), "original_prompt": data.get("original_prompt", "")})
        except Exception as e:
            logger.warning("Failed to read job file %s: %s", job_id, e)
    if job_id not in _JOBS:
        return jsonify({"error": "Unknown job id."}), 404
    _, result_text, original_prompt = _JOBS[job_id]
    if result_text is None:
        return jsonify({"status": "pending"}), 202
    return jsonify({"status": "complete", "result_markdown": result_text, "original_prompt": original_prompt})

@app.get("/jobs")
def list_jobs() -> Any:
    items = []
    try:
        for p in JOBS_DIR.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            items.append({
                "job_id": p.stem,
                "status": d.get("status", "unknown"),
                "last_update": d.get("last_update"),
                "step": d.get("step"),
                "progress": d.get("progress"),
            })
        items.sort(key=lambda x: (x.get("last_update") or 0), reverse=True)
    except Exception as e:
        logger.warning("Failed to list jobs: %s", e)
    return jsonify(items)

@app.get("/jobs/<job_id>")
def get_job(job_id: str) -> Any:
    p = _job_path(job_id)
    if not p.exists():
        return jsonify({"error": "Unknown job id."}), 404
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return jsonify(data)
    except Exception as e:
        logger.warning("Failed to read job %s: %s", job_id, e)
        return jsonify({"error": "Corrupt job file."}), 500

# -----------------------------------------------------------------------------
# TEMPLATE with HTML / CSS / JS
# -----------------------------------------------------------------------------
TEMPLATE = r"""
... (existing HTML/CSS/JS template goes here unchanged) ...
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","5000")), debug=True)