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
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple

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

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

USER_PROFILE_PATH = PROJECT_ROOT / "user_ethics_profile.json"
LATEST_SYNTHESIS_PATH = PROJECT_ROOT / "latest_synthesis.txt"
SECRET_TOKEN = os.environ.get("SECRET_TOKEN")
if not SECRET_TOKEN:
    logger.warning("SECRET_TOKEN not set; /start will reject requests without a valid token.")


# -----------------------------------------------------------------------------
# Scenario sanitizer
# -----------------------------------------------------------------------------
def sanitize_scenario(text: str) -> str:
    """
    Remove control characters (ASCII 0-31 except common whitespace) and DEL, then trim.
    Keeps visible punctuation/quotes intact.
    """
    return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text or '').strip()

# Simulated job duration (seconds). Set to 300 in real runs.
SIMULATED_DURATION_SEC: int = int(os.getenv("SIMULATED_DURATION_SEC", "300"))


# -----------------------------------------------------------------------------
# Mock data & job store (replace with real data/back end later)
# -----------------------------------------------------------------------------
MFQ_DIMENSIONS = [
    "care_harm",
    "fairness_cheating",
    "loyalty_betrayal",
    "authority_subversion",
    "purity_degradation",
]

# MFQ-30 item mean profiles (1–6 scale) for illustrative U.S. ideologies.
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

# Default worldview used server-side and passed into the template
DEFAULT_WORLDVIEW = "Moderates (US)"

# Simple in-memory job registry for demo
_JOBS: Dict[str, Tuple[float, Optional[str], str]] = {}
# job_id -> (ready_at_epoch, result_text_or_None, original_prompt)


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
    """Run the backend pipeline in a thread: write scenario/profile, call the agent, collect output."""
    ready_at = time.time() + SIMULATED_DURATION_SEC
    _JOBS[job_id] = (ready_at, None, prompt)

    def _worker() -> None:
        try:
            # 1) Persist scenario through the builder (non-interactive mode)
            subprocess.run(
                [
                    "python",
                    "scenario_builder_general.py",
                    "--scenario",
                    prompt,
                ],
                cwd=str(PROJECT_ROOT),
                check=True,
            )

            # 2) Persist active MFQ to user_ethics_profile.json (single-job only)
            # TODO: later, make this per-job to avoid global overwrite when adding concurrency.
            USER_PROFILE_PATH.write_text(
                json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # 3) Run the full synthesis pipeline end-to-end
            subprocess.run(
                ["python", "ethics_synthesis_agent.py"], cwd=str(PROJECT_ROOT), check=True
            )

            # 4) Read the final synthesis and publish it to the job store
            if LATEST_SYNTHESIS_PATH.exists():
                synthesized = LATEST_SYNTHESIS_PATH.read_text(encoding="utf-8").strip()
            else:
                synthesized = (
                    "**Pipeline error**\n\n"
                    "The backend did not produce `latest_synthesis.txt`. Please check logs."
                )
        except subprocess.CalledProcessError as e:
            synthesized = (
                "**Pipeline failed**\n\n"
                f"Exit code: {e.returncode}"
            )
        except Exception as e:  # any other failure
            synthesized = f"**Unexpected error**\n\n{e}"
        finally:
            _JOBS[job_id] = (ready_at, synthesized, prompt)
            logger.info("Job %s completed.", job_id)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def index() -> str:
    """Serve the main page with embedded HTML/CSS/JS."""
    # Render all in one template for now; later we can split into static files.
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
    # Naive heuristic: split on ., !, ?
    count = 0
    for ch in text:
        if ch in ".!?":
            count += 1
    return max(1, count) if text.strip() else 0


@app.post("/start")
def start() -> Any:
    payload: Dict[str, Any] = request.get_json(force=True, silent=False) or {}
    token = request.headers.get("X-EP-Token") or (payload.get("token") if isinstance(payload, dict) else None)
    if not token or not SECRET_TOKEN or token != SECRET_TOKEN:
        return jsonify({"error": "Unauthorized."}), 401
    scenario = sanitize_scenario(payload.get("scenario") or "")
    worldview = (payload.get("worldview") or DEFAULT_WORLDVIEW).strip()
    override_profile = payload.get("mfq_profile") if isinstance(payload.get("mfq_profile"), dict) else None
    active_profile = _active_profile_from(worldview, override_profile)

    if not scenario:
        return jsonify({"error": "Scenario text is required."}), 400

    # Server-side validation mirrors front-end limits
    if _count_words(scenario) > 80:
        return jsonify({"error": "Please keep the scenario to 80 words or fewer."}), 400

    if _count_sentences_naive(scenario) > 5:
        return jsonify({"error": "Please use five sentences or fewer."}), 400

    job_id = uuid.uuid4().hex
    _start_background_job(job_id, scenario, active_profile)

    logger.info(
        "Started job %s (worldview=%s, %d words)", job_id, worldview, _count_words(scenario)
    )

    return jsonify({
        "job_id": job_id,
        "eta_seconds": SIMULATED_DURATION_SEC,
    })


@app.get("/status/<job_id>")
def status(job_id: str) -> Any:
    if job_id not in _JOBS:
        return jsonify({"error": "Unknown job id."}), 404
    ready_at, result, _ = _JOBS[job_id]
    now = time.time()
    remaining = max(0, int(round(ready_at - now))) if result is None else 0
    return jsonify({
        "status": "complete" if result is not None else "pending",
        "eta_seconds": remaining,
    })


@app.get("/result/<job_id>")
def result(job_id: str) -> Any:
    if job_id not in _JOBS:
        return jsonify({"error": "Unknown job id."}), 404
    _, result_text, original_prompt = _JOBS[job_id]
    if result_text is None:
        return jsonify({"status": "pending"}), 202
    return jsonify({
        "status": "complete",
        "result_markdown": result_text,
        "original_prompt": original_prompt,
    })


# -----------------------------------------------------------------------------
# Template (HTML/CSS/JS)
# -----------------------------------------------------------------------------
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
        --accent: #7cf2ff; /* neon cyan */
        --accent-2: #ff7cd9; /* neon pink */
        --accent-3: #a2ff7c; /* neon green */
        --danger: #ff6b6b;
        --ok: #6bffb0;
      }
      html, body { height: 100%; }
      body {
        margin: 0; background: var(--bg); color: var(--fg);
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
      select, textarea, input, button {
        background: #0d0d0d; color: var(--fg); border: 1px solid #222;
        border-radius: 8px; padding: 10px 12px; font-size: 1rem; width: 100%;
        outline: none;
      }
      textarea { min-height: 120px; resize: vertical; }
      button.primary { background: linear-gradient(135deg, #1a1a1a, #111);
        border: 1px solid #333; cursor: pointer; }
      button.primary:hover { border-color: #444; }
      button:disabled { opacity: 0.5; cursor: not-allowed; }

      .chart-wrap { background: #0a0a0a; border: 1px solid #1a1a1a; padding: 16px;
        border-radius: 12px; }

      .hint { color: var(--muted); font-size: 0.9rem; }
      .error { color: var(--danger); margin-top: 6px; }
      .success { color: var(--ok); margin-top: 6px; }

      /* Collapsible confirmation */
      .collapsed { display: none; }
      .confirm-box { text-align: center; padding: 16px; border: 1px dashed #333;
        border-radius: 12px; }
      .confirm-actions { display: flex; gap: 12px; justify-content: center; }

      /* Circular countdown timer */
      .timer { display: none; justify-content: center; align-items: center;
        padding: 24px; }
      .timer.visible { display: flex; }
      .timer svg { width: 200px; height: 200px; }
      .timer .time-text { position: absolute; font-size: 1.6rem; }
      .timer-wrap { position: relative; }

      /* Results */
      .result { display: none; }
      .result.visible { display: block; }
      .result .faint-prompt { opacity: 0.35; font-style: italic; white-space: pre-wrap;
        border-left: 3px solid #222; padding-left: 10px; margin-bottom: 12px; }
      .result .response { white-space: pre-wrap; }
      /* Long-response ergonomics */
      .result .actions { position: sticky; top: 8px; display: flex; gap: 8px; justify-content: flex-end; margin-bottom: 12px; z-index: 5; }
      .result .actions button { padding: 8px 10px; border-radius: 8px; border: 1px solid #333; background: #0f0f0f; color: var(--fg); cursor: pointer; }
      .result .actions button:hover { border-color: #444; }
      .result .response { max-width: 80ch; margin: 0 auto; font-size: 1.05rem; line-height: 1.6; }
      .result .faint-prompt { max-width: 80ch; margin: 0 auto 12px auto; }
      .back-to-top { position: fixed; right: 24px; bottom: 24px; padding: 10px 12px; border-radius: 999px; border: 1px solid #333; background: #0f0f0f; color: var(--fg); cursor: pointer; display: none; }
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

      <!-- MFQ Chart + Worldview Toggle -->
      <section class="panel">
        <div class="row">
          <div class="col">
            <div class="label">Worldview</div>
            <select id="worldview"></select>
            <div class="hint">MFQ-30 uses a 1–6 Likert scale. These are illustrative U.S. ideology-based averages (item means); real norms vary by demographics.</div>
          </div>
          <div class="col chart-wrap">
            <canvas id="mfqChart"></canvas>
          </div>
        </div>
      </section>

      <!-- Scenario Entry -->
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
            <textarea id="scenario" placeholder="Describe the dilemma. Keep it brief and focused.\n\nContext limits: five sentences or fewer; 80 words max. This helps agents stay on-task."></textarea>
            <div class="hint"><span id="wordCount">0</span>/80 words</div>
            <div id="entryError" class="error" style="display:none"></div>
          </div>
        </div>
        <div style="margin-top: 12px; text-align: right;">
          <button id="submitEntry" class="primary" disabled>Continue</button>
        </div>
      </section>

      <!-- Confirmation step (collapses the entry) -->
      <section id="confirmPanel" class="panel collapsed">
        <div class="confirm-box">
          <p>Are you sure these settings are correct and you're ready to see the first response?</p>

          <div style="margin-bottom: 12px; max-width: 360px; margin-left: auto; margin-right: auto;">
            <div class="label">Access token</div>
            <input id="tokenInput" type="password" placeholder="Enter access token" autocomplete="off" />
            <div id="tokenError" class="error" style="display:none"></div>
          </div>

          <div class="confirm-actions">
            <button id="backBtn">Back</button>
            <button id="startBtn" class="primary">Start</button>
          </div>
        </div>
      </section>

      <!-- Timer -->
      <section id="timerPanel" class="panel timer">
        <div class="timer-wrap">
          <svg viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="54" stroke="#222" stroke-width="8" fill="none"/>
            <circle id="progressCircle" cx="60" cy="60" r="54" stroke="url(#grad)" stroke-width="8"
              stroke-linecap="round" fill="none" stroke-dasharray="339.292" stroke-dashoffset="0"/>
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

      <!-- Results -->
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
      // ---------------------- Data bootstrapped from Flask ----------------------
      const MFQ_DIMENSIONS = {{ mfq_dimensions | tojson }};
      const NORM_PROFILES = {{ norm_profiles | tojson }};
      const DEFAULT_WORLDVIEW = {{ default_worldview | tojson }};
      const SIM_DURATION = {{ simulated_duration_sec | tojson }}; // seconds

      // ------------------------------ MFQ Chart --------------------------------
      const worldviewSelect = document.getElementById('worldview');
      const ctx = document.getElementById('mfqChart');

      // Improved labels for MFQ dimensions
      const DIM_LABELS = {
        care_harm: 'Care/Harm',
        fairness_cheating: 'Fairness/Reciprocity',
        loyalty_betrayal: 'Ingroup/Loyalty',
        authority_subversion: 'Authority/Respect',
        purity_degradation: 'Purity/Sanctity',
      };

      // Populate worldview options
      Object.keys(NORM_PROFILES).forEach(name => {
        const opt = document.createElement('option');
        opt.value = name; opt.textContent = name;
        if (name === DEFAULT_WORLDVIEW) opt.selected = true;
        worldviewSelect.appendChild(opt);
      });

      function profileToArray(name) {
        const obj = NORM_PROFILES[name] || NORM_PROFILES[DEFAULT_WORLDVIEW];
        return MFQ_DIMENSIONS.map(k => obj[k] ?? 0);
      }

      const neonPalette = [
        'rgba(124, 242, 255, 0.9)', // cyan
        'rgba(162, 255, 124, 0.9)', // green
        'rgba(255, 124, 217, 0.9)', // pink
        'rgba(255, 215, 124, 0.9)', // amber
        'rgba(124, 148, 255, 0.9)', // indigo
      ];

      const chart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: MFQ_DIMENSIONS.map(k => DIM_LABELS[k] || k),
          datasets: [{
            label: 'Moral Foundations (1–6 item mean)',
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
            y: { beginAtZero: true, min: 1, max: 6, ticks: { color: '#bbb' }, grid: { color: '#111' } },
          }
        }
      });

      worldviewSelect.addEventListener('change', () => {
        chart.data.datasets[0].data = profileToArray(worldviewSelect.value);
        chart.update();
      });

      // --------------------------- Scenario Input ------------------------------
      const preset = document.getElementById('preset');
      const scenarioEl = document.getElementById('scenario');
      const submitEntry = document.getElementById('submitEntry');
      const wordCount = document.getElementById('wordCount');
      const entryError = document.getElementById('entryError');

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
        const m = text.match(/[\.!?]/g);
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
        submitEntry.disabled = !scenarioEl.value.trim() || tooManyWords || tooManySentences;
      }

      scenarioEl.addEventListener('input', updateCounts);
      updateCounts();

      // -------------------------- Confirm & Start ------------------------------
      const entryPanel = document.getElementById('entryPanel');
      const confirmPanel = document.getElementById('confirmPanel');
      const backBtn = document.getElementById('backBtn');
      const startBtn = document.getElementById('startBtn');
      const tokenInput = document.getElementById('tokenInput');
      const tokenError = document.getElementById('tokenError');

      submitEntry.addEventListener('click', () => {
        entryPanel.classList.add('collapsed');
        confirmPanel.classList.remove('collapsed');
      });

      backBtn.addEventListener('click', () => {
        confirmPanel.classList.add('collapsed');
        entryPanel.classList.remove('collapsed');
      });

      // ------------------------------ Timer UI --------------------------------
      const timerPanel = document.getElementById('timerPanel');
      const progressCircle = document.getElementById('progressCircle');
      const timeText = document.getElementById('timeText');
      const CIRCUMFERENCE = 2 * Math.PI * 54; // r=54 as in SVG

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

        // Poll status (every 2s). If complete, show result immediately.
        statusInterval = setInterval(async () => {
          const res = await fetch(`/status/${jobId}`);
          const data = await res.json();
          if (data.status === 'complete') {
            clearInterval(statusInterval);
            clearInterval(countdownInterval);
            showResult(jobId);
          } else if (typeof data.eta_seconds === 'number') {
            // Keep the timer honest if the back end has a better ETA
            setProgress(data.eta_seconds, total);
          }
        }, 2000);
      }

      // ------------------------------- Run ------------------------------------
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
        // Copy the plain-text version of the response
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
        // Create a simple markdown export of the response
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

      // Back to top visibility & behavior
      window.addEventListener('scroll', () => {
        if (window.scrollY > 200) backToTopBtn.classList.add('visible');
        else backToTopBtn.classList.remove('visible');
      });
      backToTopBtn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));

      function showResult(jobId) {
        (async () => {
          const res = await fetch(`/result/${jobId}`);
          const data = await res.json();
          if (data.status !== 'complete') return; // should be complete

          timerPanel.classList.remove('visible');
          resultPanel.classList.add('visible');
          faintPrompt.textContent = data.original_prompt || '';

          // Render markdown-ish minimal (bold + newlines) without external libs
          const md = (data.result_markdown || '')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1<\/strong>')
            .replace(/\n/g, '<br/>');
          response.innerHTML = md;
        })();
      }

      startBtn.addEventListener('click', async () => {
        // Freeze config and kick off back end
        confirmPanel.classList.add('collapsed');

        const token = (tokenInput && tokenInput.value || '').trim();
        if (!token) {
          if (tokenError) {
            tokenError.style.display = 'block';
            tokenError.textContent = 'Access token required.';
          }
          confirmPanel.classList.remove('collapsed');
          return;
        }
        if (tokenError) {
          tokenError.style.display = 'none';
          tokenError.textContent = '';
        }

        const currentData = chart.data.datasets[0].data;
        const profileObj = {};
        MFQ_DIMENSIONS.forEach((k, i) => { profileObj[k] = Number(currentData[i] ?? 0); });

        const body = {
          scenario: scenarioEl.value.trim(),
          worldview: worldviewSelect.value,
          mfq_profile: profileObj,
        };

        const res = await fetch('/start', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-EP-Token': token
          },
          body: JSON.stringify(body)
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({error:'Unknown error'}));
          entryError.style.display = 'block';
          entryError.textContent = err.error || 'Unable to start. Please try again.';
          // Go back to entry so user can fix
          entryPanel.classList.remove('collapsed');
          return;
        }

        const { job_id, eta_seconds } = await res.json();
        startCountdown(job_id, typeof eta_seconds === 'number' ? eta_seconds : SIM_DURATION);
      });
    </script>
  </body>
</html>
"""


if __name__ == "__main__":  # Dev server
    app.run(host="127.0.0.1", port=5000, debug=True)
