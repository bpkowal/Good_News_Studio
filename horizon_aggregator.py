"""
horizon_aggregator.py
---------------------

A tiny helper for *time-bounded utilitarian* calculations.

The idea
========
Many ethical trade-offs hinge on **when** a benefit or burden occurs
as much as on **how large** it is.  This module provides:

1.  **A rough horizon estimator** -maps heuristic temporal tags
    (``"immediate"``, ``"short_term"``, …) to a cut-off distance.
2.  **Diminishing-value kernels** - flat, exponential, or hyperbolic.
3.  **An aggregator** - sums only those outcomes that fall *within* the
    horizon, applying the chosen kernel.
4.  **A convenience wrapper** that returns a JSON-friendly summary
    you can embed directly in an LLM prompt.

This keeps the number-crunching on the Python side while letting the
language model focus on higher-level moral reasoning.

Usage
=====
```python
from horizon_aggregator import compute_summary

summary = compute_summary(
    temporal_tags=["short_term", "long_term"],
    outcome_values=[+10, -2, +1],
    outcome_distances=[0, 48, 8760],      # hours in the future
    kernel="hyperbolic",                  # or "flat" / "exponential"
    k=0.02                                # optional kernel kwargs
)
print(summary)
```
"""

from __future__ import annotations

import math
from typing import Dict, Sequence
import difflib

# ---------------------------------------------------------------------------
# 1. Horizon estimation
# ---------------------------------------------------------------------------

# Very crude mapping.  Tweak or extend as needed for your domain.
_TAG_TO_HOURS: Dict[str, float] = {
    "immediate": 0,
    "short_term": 24,          # ~1 day
    "medium_term": 24 * 7,     # ~1 week
    "long_term": 24 * 30,      # ~1 month
    "very_long_term": 24 * 365 # ~1 year
}

# -----------------------------------------------------------------------
#  Additional semantic aliases so we can recognise tags that *suggest*
#  temporal scope without using the exact canonical words.
# -----------------------------------------------------------------------
_SYNONYM_TO_CANONICAL = {
    # Short‑ish horizon cues
    "time preference": "short_term",
    "impatience": "short_term",
    "opportunity cost": "short_term",
    "near term": "short_term",
    "near‑term": "short_term",
    "imminent": "immediate",

    # Longer horizon cues
    "discounted cash flow": "long_term",
    "dcf": "long_term",
    "patience": "long_term",
    "delayed reward": "long_term",
    "future value": "long_term",
}

_ALL_KNOWN_LABELS = set(_TAG_TO_HOURS) | set(_SYNONYM_TO_CANONICAL)


def estimate_horizon(temporal_tags: Sequence[str]) -> float:
    """
    Return an *upper-bound* horizon (in hours) implied by the supplied tags.

    If no recognised tags are present we default to the longest horizon so
    that nothing is inadvertently excluded.
    """
    # Normalise inputs
    cleaned = [tag.strip().lower() for tag in temporal_tags]

    horizons: list[float] = []
    for tag in cleaned:
        # 1) Direct canonical hit
        if tag in _TAG_TO_HOURS:
            horizons.append(_TAG_TO_HOURS[tag])
            continue

        # 2) Direct synonym hit
        if tag in _SYNONYM_TO_CANONICAL:
            canon = _SYNONYM_TO_CANONICAL[tag]
            horizons.append(_TAG_TO_HOURS[canon])
            continue

        # 3) Fuzzy match against any known label
        #    We look for a high‑confidence close match (ratio ≥ 0.8)
        close = difflib.get_close_matches(tag, _ALL_KNOWN_LABELS, n=1, cutoff=0.8)
        if close:
            best = close[0]
            canon = _SYNONYM_TO_CANONICAL.get(best, best)  # map synonym→canon if needed
            horizons.append(_TAG_TO_HOURS[canon])
            continue

        # 4) Fallback: treat as maximally distant so we *include* it
        horizons.append(max(_TAG_TO_HOURS.values()))

    # Use the **maximum** horizon so the most remote tag dominates.
    return max(horizons) if horizons else max(_TAG_TO_HOURS.values())


# ---------------------------------------------------------------------------
# 2. Diminishing‑value kernels
# ---------------------------------------------------------------------------

def _flat(value: float, distance: float, **_: float) -> float:
    """No discounting at all."""
    return value


def _exponential(value: float, distance: float, decay: float = 0.01, **_: float) -> float:
    """Standard exponential discounting."""
    return value * math.exp(-decay * distance)


def _hyperbolic(value: float, distance: float, k: float = 0.015, **_: float) -> float:
    """Hyperbolic discounting (favoured in behavioural econ)."""
    return value / (1.0 + k * distance)


_KERNELS = {
    "flat": _flat,
    "exponential": _exponential,
    "hyperbolic": _hyperbolic,
}

# Named aliases expected by some older code -------------------------------
flat_kernel = _flat
exponential_kernel = _exponential
hyperbolic_kernel = _hyperbolic

# ---------------------------------------------------------------------------
# 3. Core aggregation logic
# ---------------------------------------------------------------------------

def aggregate(
    values: Sequence[float],
    distances: Sequence[float],
    horizon: float,
    *,
    kernel: str = "hyperbolic",
    **kernel_kwargs: float,
) -> float:
    """
    Sum *values* whose *distance* ≤ *horizon*, applying the chosen kernel.

    Parameters
    ----------
    values
        Magnitudes (positive for benefits, negative for harms).
    distances
        Temporal distance in **hours**.
    horizon
        Upper bound (in hours) beyond which events are ignored.
    kernel
        One of ``"flat"``, ``"exponential"``, ``"hyperbolic"``.
    **kernel_kwargs
        Extra parameters forwarded to the kernel (e.g. ``decay=0.05``).
    """
    if len(values) != len(distances):
        raise ValueError("values and distances must be of equal length")

    kernel_fn = _KERNELS.get(kernel)
    if kernel_fn is None:
        raise ValueError(f"Unknown kernel '{kernel}'. "
                         f"Choose from {', '.join(_KERNELS)}")

    total = 0.0
    for v, d in zip(values, distances):
        if d <= horizon:
            total += kernel_fn(v, d, **kernel_kwargs)
    return total


# ---------------------------------------------------------------------------
# 3b. Backward‑compatibility wrapper (legacy name expected by agents)
# ---------------------------------------------------------------------------

def horizon_limited_aggregate(
    values: Sequence[float],
    distances: Sequence[float],
    temporal_tags: Sequence[str] | None = None,
    *,
    kernel: str = "hyperbolic",
    **kernel_kwargs: float,
) -> float:
    """
    Legacy wrapper kept for older agent code.

    * If *temporal_tags* is provided we infer the horizon from them;
      otherwise we assume the **maximum** horizon so that nothing is cut off.
    * All other arguments are forwarded to :func:`aggregate`.
    """
    horizon = estimate_horizon(temporal_tags or [])
    return aggregate(
        values,
        distances,
        horizon,
        kernel=kernel,
        **kernel_kwargs,
    )


# ---------------------------------------------------------------------------
# 4. High‑level helper
# ---------------------------------------------------------------------------

def compute_summary(
    temporal_tags: Sequence[str],
    outcome_values: Sequence[float],
    outcome_distances: Sequence[float],
    *,
    kernel: str = "hyperbolic",
    **kernel_kwargs: float,
) -> Dict[str, float]:
    """
    Return a dictionary summarising horizon-limited utility.

    The result is deliberately JSON-serialisable for easy inclusion in
    prompt engineering.

    Keys
    ----
    horizon
        The inferred horizon in hours.
    kernel
        The discounting kernel used.
    total_utility
        The aggregated, horizon-limited utility.
    n_events
        How many outcome/events were provided.
    """
    horizon = estimate_horizon(temporal_tags)
    total = aggregate(
        outcome_values,
        outcome_distances,
        horizon,
        kernel=kernel,
        **kernel_kwargs,
    )
    return {
        "horizon": horizon,
        "kernel": kernel,
        "total_utility": total,
        "n_events": len(outcome_values),
    }


# Alias retained for older agent code expecting a different name
estimate_horizon_from_tags = estimate_horizon

# Convenience export
__all__ = [
    "estimate_horizon",
    "aggregate",
    "horizon_limited_aggregate",
    "compute_summary",
    "flat_kernel",
    "exponential_kernel",
    "hyperbolic_kernel",
    "estimate_horizon_from_tags",
]