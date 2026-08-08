from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EpisodicMemory:
    """Append-only storage. Retrieved rules are priors, never automatic commands."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, episode: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        episodes = []
        for line in lines:
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return episodes
