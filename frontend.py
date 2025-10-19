from __future__ import annotations

"""Front-end server for Ethical Parliament — with token/password prompt UI (confirm step)."""

import logging
import os
import time
import uuid
import threading
import json
import pathlib
import subprocess
import re
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

SIMULATED_DURATION_SEC = int(os.getenv("SIMULATED_DURATION_SEC", "150"))

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
    base = dict(NORM_PROFILES.get(worldview, NORM_PROFILES.get(DEFAULT_WORLDVIEW, {})))
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
                synthesized = "**Pipeline error** — no output"
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

def _count_words(text: str) -> int:
    return len([w for w in text.strip().split() if w])

def _count_sentences_naive(text: str) -> int:
    if not text.strip():
        return 0
    return len(re.findall(r"[.!?]", text))

@app.get("/")
def index() -> str:
    return render_template_string(
        TEMPLATE,
        secret_token=SECRET_TOKEN,
        simulated_duration_sec=SIMULATED_DURATION_SEC,
        mfq_dimensions=MFQ_DIMENSIONS,
        norm_profiles=NORM_PROFILES,
        default_worldview=DEFAULT_WORLDVIEW,
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
    if _count_words(scenario) > 80:
        return jsonify({"error": "Please limit to 80 words."}), 400
    if _count_sentences_naive(scenario) > 5:
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
                "result_markdown": d.get("result_markdown", ""),
                "original_prompt": d.get("original_prompt", ""),
            })
        except Exception as e:
            logger.warning("Failed to read result %s: %s", job_id, e)
    _, resp, prompt = _JOBS.get(job_id, (0, None, ""))
    if resp is None:
        return jsonify({"status": "pending"}), 202
    return jsonify({"status": "complete", "result_markdown": resp, "original_prompt": prompt})

@app.get("/jobs")
def list_jobs():
    items = []
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

# ----------------------------------------------------------------------------

TEMPLATE = r"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Ethical Parliament</title>
    <link rel="preconnect" href="https://cdn.jsdelivr.net" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
      :root {
        --bg: #000;
        --fg: #f2f2f2;
        --muted: #9aa0a6;
        --accent: #7cf2ff;
        --accent-2: #ff7cd9;
        --accent-3: #a2ff7c;
        --danger: #ff6b6b;
        --ok: #6bffb0;
      }
      html, body { height: 100%; }
      body {
        margin: 0;
        background: var(--bg);
        color: var(--fg);
        font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu,
          Cantarell, Noto Sans, Helvetica Neue, Arial, "Apple Color Emoji",
          "Segoe UI Emoji";
      }
      a { color: var(--accent); text-decoration: none; }
      a:hover { text-decoration: underline; }

      .wrap { max-width: 1000px; margin: 0 auto; padding: 24px; }
      header { display: flex; align-items: baseline; gap: 12px; }
      header h1 { margin: 0; font-size: 1.8rem; letter-spacing: 0.5px; }
      header .credit { margin-left: auto; font-size: 0.9rem; color: var(--muted); }

      .panel { border: 1px solid #222; border-radius: 12px; padding: 16px; }
      .panel + .panel { margin-top: 16px; }

      .row { display: flex; gap: 16px; flex-wrap: wrap; }
      .col { flex: 1 1 320px; }

      .label { color: var(--muted); font-size: 0.9rem; margin-bottom: 8px; }
      select, textarea, button, input {
        background: #0d0d0d;
        color: var(--fg);
        border: 1px solid #222;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 1rem;
        width: 100%;
        outline: none;
      }
      textarea { min-height: 120px; resize: vertical; }
      button.primary {
        background: linear-gradient(135deg, #1a1a1a, #111);
        border: 1px solid #333;
        cursor: pointer;
      }
      button.primary:hover { border-color: #444; }
      button:disabled { opacity: 0.5; cursor: not-allowed; }

      .chart-wrap {
        background: #0a0a0a;
        border: 1px solid #1a1a1a;
        padding: 16px;
        border-radius: 12px;
      }

      .hint { color: var(--muted); font-size: 0.9rem; }
      .error { color: var(--danger); margin-top: 6px; }
      .success { color: var(--ok); margin-top: 6px; }

      .collapsed { display: none; }
      .confirm-box {
        text-align: center; padding: 16px; border: 1px dashed #333;
        border-radius: 12px;
      }
      .confirm-actions {
        display: flex; gap: 12px; justify-content: center;
      }

      .timer { display: none; justify-content: center; align-items: center; padding: 24px; }
      .timer.visible { display: flex; }
      .timer svg { width: 200px; height: 200px; }
      .timer .time-text { position: absolute; font-size: 1.6rem; }
      .timer-wrap { position: relative; }

      .result { display: none; }
      .result.visible { display: block; }
      .result .faint-prompt {
        opacity: 0.35; font-style: italic; white-space: pre-wrap;
        border-left: 3px solid #222; padding-left: 10px; margin-bottom: 12px;
      }
      .result .response { white-space: pre-wrap; }
      .result .actions {
        position: sticky; top: 8px;
        display: flex; gap: 8px;
        justify-content: flex-end;
        margin-bottom: 12px; z-index: 5;
      }
      .result .actions button {
        padding: 8px 10px; border-radius: 8px;
        border: 1px solid #333; background: #0f0f0f;
        color: var(--fg); cursor: pointer;
      }
      .result .actions button:hover { border-color: #444; }
      .result .response { max-width: 80ch; margin: 0 auto; font-size: 1.05rem; line-height: 1.6; }
      .result .faint-prompt { max-width: 80ch; margin: 0 auto 12px auto; }
      .back-to-top {
        position: fixed; right: 24px; bottom: 24px;
        padding: 10px 12px; border-radius: 999px;
        border: 1px solid #333; background: #0f0f0f;
        color: var(--fg); cursor: pointer; display: none;
      }
      .back-to-top.visible { display: block; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <header>
        <h1>Ethical Parliament</h1>
        <div class="credit">
          Based on the Moral Foundations framework. Data courtesy of the
          <a href="https://moralfoundations.org" target="_blank" rel="noreferrer">MFQ project</a>.
        </div>
      </header>

      <section class="panel">
        <div class="row">
          <div class="col">
            <div class="label">Worldview</div> <select id="worldview"></select>
            <div class="hint">Select a normative profile to visualize moral weights.</div>
          </div>
          <div class="col chart-wrap">
            <canvas id="mfqChart"></canvas>
          </div>
        </div>
      </section>

      <section id="entryPanel" class="panel">
        <div class="row">
          <div class="col">
            <div class="label">Pick a classic dilemma (optional)</div>
            <select id="preset">
              <option value="">— None —</option>
              <option value="trolley">Trolley problem (classic switch)</option>
              <option value="organ">Organ donation (transplant sacrifice)</option>
              <option value="triage">Patient triage (limited ventilators)</option>
              <option value="pig">Genetically contented pigs (ethics of eating)</option>
            </select>
          </div>
          <div class="col">
            <div class="label">Your scenario (≤ 80 words, ≤ 5 sentences)</div>
            <textarea id="scenario" placeholder="Describe the dilemma. Keep it brief and focused."></textarea>
            <div class="hint"><span id="wordCount">0</span>/80 words</div>
            <div id="entryError" class="error" style="display:none"></div>
          </div>
        </div>
        <div style="margin-top: 12px; text-align: right;">
          <button id="submitEntry" class="primary" disabled>Continue</button>
        </div>
      </section>

      <section id="confirmPanel" class="panel collapsed">
        <div class="confirm-box">
          <p>Are you sure these settings are correct and you're ready to see the first response?</p>
          <div style="margin-top:12px;">
            <div class="label">Access token / password</div>
            <input type="password" id="accessToken" placeholder="Enter password/token" />
            <div id="tokenError" class="error" style="display:none"></div>
          </div>
          <div class="confirm-actions">
            <button id="backBtn">Back</button>
            <button id="startBtn" class="primary">Start</button>
          </div>
        </div>
      </section>

      <section id="timerPanel" class="panel timer">
        <div class="timer-wrap">
          <svg viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="54" stroke="#222" stroke-width="8" fill="none"/>
            <circle id="progressCircle" cx="60" cy="60" r="54"
              stroke="url(#grad)" stroke-width="8"
              stroke-linecap="round" fill="none"
              stroke-dasharray="339.292" stroke-dashoffset="0" />
            <defs>
              <linearGradient id="grad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="var(--accent)" />
                <stop offset="100%" stop-color="var(--accent-2)" />
              </linearGradient>
            </defs>
          </svg>
          <div class="time-text" id="timeText">—:—</div>
        </div>
      </section>

      <section id="resultPanel" class="panel result">
        <div class="actions">
          <button id="togglePromptBtn">Hide prompt</button>
          <button id="copyBtn">Copy response</button>
          <button id="downloadBtn">Download .md</button>
        </div>
        <div class="faint-prompt" id="faintPrompt"></div>
        <div class="response" id="response"></div>
      </section>
      <button id="backToTop" class="back-to-top">Top</button>
    </div>

    <script>
      const MFQ_DIMENSIONS = {{ mfq_dimensions | tojson }};
      const NORM_PROFILES = {{ norm_profiles | tojson }};
      const DEFAULT_WORLDVIEW = {{ default_worldview | tojson }};
      const SIM_DURATION = {{ simulated_duration_sec | tojson }};

      const worldviewSelect = document.getElementById('worldview');
      const ctx = document.getElementById('mfqChart');

      Object.keys(NORM_PROFILES).forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        if (name === DEFAULT_WORLDVIEW) opt.selected = true;
        worldviewSelect.appendChild(opt);
      });

      function profileToArray(name) {
        const obj = NORM_PROFILES[name] || NORM_PROFILES[DEFAULT_WORLDVIEW];
        return MFQ_DIMENSIONS.map(k => obj[k] ?? 0);
      }

      const neonPalette = [
        'rgba(124, 242, 255, 0.9)',
        'rgba(162, 255, 124, 0.9)',
        'rgba(255, 124, 217, 0.9)',
        'rgba(255, 215, 124, 0.9)',
        'rgba(124, 148, 255, 0.9)',
      ];

      const chart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: MFQ_DIMENSIONS.map(s => s.replace('_', ' ')),
          datasets: [{
            label: 'Moral Foundations (0–5)',
            data: profileToArray(DEFAULT_WORLDVIEW),
            backgroundColor: neonPalette,
            borderColor: neonPalette.map(c => c.replace('0.9', '1.0')),
            borderWidth: 1
          }]
        },
        options: {
          plugins: {
            legend: { labels: { color: '#ddd' } },
            tooltip: { enabled: true },
          },
          scales: {
            x: { ticks: { color: '#bbb' }, grid: { color: '#111' } },
            y: { beginAtZero: true, max: 5, ticks: { color: '#bbb' }, grid: { color: '#111' } },
          }
        }
      });

      worldviewSelect.addEventListener('change', () => {
        chart.data.datasets[0].data = profileToArray(worldviewSelect.value);
        chart.update();
      });

      const preset = document.getElementById('preset');
      const scenarioEl = document.getElementById('scenario');
      const submitEntry = document.getElementById('submitEntry');
      const accessTokenEl = document.getElementById('accessToken');
      const wordCount = document.getElementById('wordCount');
      const entryError = document.getElementById('entryError');
      const tokenError = document.getElementById('tokenError');

      const PRESETS = {
        trolley: "A runaway trolley will kill five workers unless I divert it onto a sidetrack where one worker will die instead. I can pull a lever to divert it.",
        organ: "A surgeon has five patients needing organs and a healthy stranger whose organs could save them all. Should the surgeon sacrifice the stranger to save five?",
        triage: "A hospital has three patients and one ventilator during a crisis: a young nurse, an elderly scholar, and a parent of two. Who should receive the ventilator?",
        pig: "Engineered pigs are raised to strongly prefer being eaten if well-treated. Is it more ethical to eat them than standard livestock?",
      };

      preset.addEventListener('change', () => {
        const key = preset.value.trim();
        if (key && PRESETS[key]) {
          scenarioEl.value = PRESETS[key];
          updateCounts();
        }
      });

      function countWords(text) {
        return (text.trim().match(/\S+/g) || []).length;
      }

      function countSentences(text) {
        const m = text.match(/[.!?]/g);
        if (!text.trim()) return 0;
        return Math.max(1, m ? m.length : 0);
      }

      function updateCounts() {
        const words = countWords(scenarioEl.value);
        wordCount.textContent = String(words);
        const tooManyWords = words > 80;
        const tooManySentences = countSentences(scenarioEl.value) > 5;
        if (tooManyWords) {
          entryError.style.display = 'block';
          entryError.textContent = 'Please keep to 80 words or fewer.';
        } else if (tooManySentences) {
          entryError.style.display = 'block';
          entryError.textContent = 'Please use five sentences or fewer.';
        } else {
          entryError.style.display = 'none';
          entryError.textContent = '';
        }
        submitEntry.disabled =
          !scenarioEl.value.trim() ||
          tooManyWords ||
          tooManySentences;
      }

      scenarioEl.addEventListener('input', updateCounts);
      updateCounts();

      const entryPanel = document.getElementById('entryPanel');
      const confirmPanel = document.getElementById('confirmPanel');
      const backBtn = document.getElementById('backBtn');
      const startBtn = document.getElementById('startBtn');

      submitEntry.addEventListener('click', () => {
        entryPanel.classList.add('collapsed');
        confirmPanel.classList.remove('collapsed');
        accessTokenEl.focus();
      });

      backBtn.addEventListener('click', () => {
        confirmPanel.classList.add('collapsed');
        entryPanel.classList.remove('collapsed');
      });

      const timerPanel = document.getElementById('timerPanel');
      const progressCircle = document.getElementById('progressCircle');
      const timeText = document.getElementById('timeText');
      const CIRCUMFERENCE = 2 * Math.PI * 54;

      let countdownInterval = null;
      let statusInterval = null;

      function setProgress(remaining, total) {
        const ratio = Math.max(0, Math.min(1, remaining / total));
        const offset = CIRCUMFERENCE * (1 - ratio);
        progressCircle.setAttribute('stroke-dasharray', String(CIRCUMFERENCE));
        progressCircle.setAttribute('stroke-dashoffset', String(offset));
        const mm = Math.floor(remaining / 60);
        const ss = remaining % 60;
        timeText.textContent = `${String(mm).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
      }

      function startCountdown(jobId, total) {
        let remaining = total;
        timerPanel.classList.add('visible');
        setProgress(remaining, total);

        countdownInterval = setInterval(() => {
          remaining = Math.max(0, remaining - 1);
          setProgress(remaining, total);
          if (remaining <= 0) {
            clearInterval(countdownInterval);
          }
        }, 1000);

        statusInterval = setInterval(async () => {
          const res = await fetch(`/status/${jobId}`);
          const data = await res.json();
          if (data.status === 'complete') {
            clearInterval(statusInterval);
            clearInterval(countdownInterval);
            showResult(jobId);
          } else if (typeof data.eta_seconds === 'number') {
            setProgress(data.eta_seconds, total);
          }
        }, 2000);
      }

      const resultPanel = document.getElementById('resultPanel');
      const faintPrompt = document.getElementById('faintPrompt');
      const response = document.getElementById('response');
      const togglePromptBtn = document.getElementById('togglePromptBtn');
      const copyBtn = document.getElementById('copyBtn');
      const downloadBtn = document.getElementById('downloadBtn');
      const backToTopBtn = document.getElementById('backToTop');

      togglePromptBtn.addEventListener('click', () => {
        const isHidden = faintPrompt.style.display === 'none';
        faintPrompt.style.display = isHidden ? '' : 'none';
        togglePromptBtn.textContent = isHidden ? 'Hide prompt' : 'Show prompt';
      });

      copyBtn.addEventListener('click', async () => {
        const temp = document.createElement('div');
        temp.innerHTML = response.innerHTML.replace(/<br\/>/g, '\n');
        const text = temp.textContent || temp.innerText || '';
        try {
          await navigator.clipboard.writeText(text);
          copyBtn.textContent = 'Copied!';
          setTimeout(() => (copyBtn.textContent = 'Copy response'), 1200);
        } catch (e) {
          alert('Copy failed.');
        }
      });

      downloadBtn.addEventListener('click', () => {
        const md = response.innerHTML
          .replace(/<strong>(.*?)<\/strong>/g, '**$1**')
          .replace(/<br\s*\/>/g, '\n')
          .replace(/<[^>]+>/g, '');
        const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'ethical_parliament_response.md';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      });

      window.addEventListener('scroll', () => {
        if (window.scrollY > 200) backToTopBtn.classList.add('visible');
        else backToTopBtn.classList.remove('visible');
      });
      backToTopBtn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));

      function showResult(jobId) {
        (async () => {
          const res = await fetch(`/result/${jobId}`);
          const data = await res.json();
          if (data.status !== 'complete') return;

          timerPanel.classList.remove('visible');
          resultPanel.classList.add('visible');
          faintPrompt.textContent = data.original_prompt || '';

          const md = (data.result_markdown || '')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br/>');
          response.innerHTML = md;
        })();
      }

      startBtn.addEventListener('click', async () => {
        const token = accessTokenEl.value.trim();
        if (!token) {
          tokenError.style.display = 'block';
          tokenError.textContent = 'Token/password is required.';
          return;
        }
        tokenError.style.display = 'none';

        accessTokenEl.disabled = true;
        startBtn.disabled = true;

        const body = {
          scenario: scenarioEl.value.trim(),
          worldview: worldviewSelect.value,
          token: token,
        };

        const res = await fetch('/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ error: 'Unknown error' }));
          entryError.style.display = 'block';
          entryError.textContent = err.error || 'Unable to start. Please try again.';
          confirmPanel.classList.add('collapsed');
          entryPanel.classList.remove('collapsed');
          accessTokenEl.disabled = false;
          startBtn.disabled = false;
          return;
        }

        const { job_id, eta_seconds } = await res.json();
        startCountdown(job_id, typeof eta_seconds === 'number' ? eta_seconds : SIM_DURATION);
      });
    </script>

  </body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)