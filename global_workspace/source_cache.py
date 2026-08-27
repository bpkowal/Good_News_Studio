from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def scenario_cache_fingerprint(scenario_id: str, scenario_path: Path | None) -> str:
    """Return a stable fingerprint for the current scenario payload."""
    if scenario_path is not None and scenario_path.exists():
        payload = scenario_path.read_bytes()
    else:
        payload = scenario_id.encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def build_source_cache_key(
    prompt_version: str,
    query: str,
    scenario_id: str,
    scenario_path: Path | None,
    *extra_parts: str,
) -> str:
    parts = [
        str(prompt_version).strip(),
        str(scenario_id).strip(),
        scenario_cache_fingerprint(scenario_id, scenario_path),
        " ".join(str(query).split()),
    ]
    for part in extra_parts:
        normalized = " ".join(str(part).split())
        if normalized:
            parts.append(normalized)
    return "\n".join(parts)
