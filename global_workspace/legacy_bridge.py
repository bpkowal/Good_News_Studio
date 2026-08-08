from __future__ import annotations

import argparse
import base64
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


AGENT_MODULES = {
    "utilitarian": "utilitarian_agent_p",
    "deontological": "deontological_agent_p",
    "virtue": "virtue_ethics_agent_p",
    "care": "care_ethics_agent_p",
    "rawlsian": "rawlsian_ethics_agent_p",
}
RESPONSE_MARKER = "WORKSPACE_RESPONSE_B64="


@dataclass(slots=True)
class LegacyConsultation:
    testimonies: dict[str, str]
    errors: dict[str, str]


def _child_consult(agent: str, scenario_path: Path, max_tokens: int) -> int:
    if agent not in AGENT_MODULES:
        raise ValueError(f"Unknown original agent: {agent}")
    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    question = str(data.get("ethical_question", "")).strip()
    if not question:
        raise ValueError("Scenario has no ethical_question")
    module = importlib.import_module(AGENT_MODULES[agent])
    response = module.respond_to_query(
        query=question,
        scenario_id=scenario_path.stem,
        scenario_path=scenario_path,
        max_tokens=max_tokens,
    )
    encoded = base64.b64encode(str(response).encode("utf-8")).decode("ascii")
    print(f"{RESPONSE_MARKER}{encoded}")
    return 0


def _decode_response(stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESPONSE_MARKER):
            payload = line[len(RESPONSE_MARKER):]
            return base64.b64decode(payload).decode("utf-8").strip()
    raise ValueError("Original agent produced no bridge response marker")


def consult_original_agents(
    scenario_path: Path,
    *,
    agents: tuple[str, ...] = tuple(AGENT_MODULES),
    max_tokens: int = 180,
    timeout_seconds: float = 600.0,
) -> LegacyConsultation:
    testimonies: dict[str, str] = {}
    errors: dict[str, str] = {}
    for agent in agents:
        print(f"Consulting original {agent} agent...", flush=True)
        command = [
            sys.executable,
            "-m",
            "global_workspace.legacy_bridge",
            "--agent",
            agent,
            "--scenario",
            str(scenario_path),
            "--max-tokens",
            str(max_tokens),
        ]
        try:
            child_env = os.environ.copy()
            # Avoid remote HEAD requests when the embedding model is already cached.
            child_env.setdefault("HF_HUB_OFFLINE", "1")
            child_env.setdefault("TRANSFORMERS_OFFLINE", "1")
            child_env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parent.parent,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip().splitlines()
                errors[agent] = detail[-1][:500] if detail else f"exit code {completed.returncode}"
                continue
            testimony = _decode_response(completed.stdout)
            if testimony:
                testimonies[agent] = testimony
            else:
                errors[agent] = "empty response"
        except (subprocess.TimeoutExpired, ValueError, UnicodeError) as exc:
            errors[agent] = str(exc)[:500]
    return LegacyConsultation(testimonies=testimonies, errors=errors)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=tuple(AGENT_MODULES))
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    child_args = _parse_args()
    raise SystemExit(_child_consult(child_args.agent, child_args.scenario.resolve(), child_args.max_tokens))
