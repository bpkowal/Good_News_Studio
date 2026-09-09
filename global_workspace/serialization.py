"""Safe JSON persistence helpers for workspace traces and checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    pretty: bool,
) -> int:
    """Write JSON through a same-directory temporary file, then replace.

    Checkpoints use compact JSON because they are rewritten during a run. The
    final audit remains indented for inspection. Both modes contain the same
    complete payload and become visible only after a successful write.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                indent=2 if pretty else None,
                separators=None if pretty else (",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        byte_count = temporary_path.stat().st_size
        os.replace(temporary_path, destination)
        temporary_path = None
        return byte_count
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
