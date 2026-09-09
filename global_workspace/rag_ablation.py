"""Select committed worlds and compare RAG / no-RAG / skip-original arms."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from global_workspace.framework_retrieval import (
    EvidenceThresholds,
    load_corpus_passages,
    retrieve_framework_evidence,
)
from global_workspace.legacy_bridge import AGENT_MODULES
from global_workspace.retrieval_trace import (
    annotate_retrieval_use,
    serialize_retrieval_result,
    specialist_cycle_text,
)
from global_workspace_pipeline import save_problem_framing_cache


ROOT = Path(__file__).resolve().parent.parent

AGENT_RETRIEVAL_SPECS: dict[str, dict[str, Any]] = {
    "utilitarian": {
        "corpus_dir": ROOT / "utilitarian_corpus",
        "thresholds": EvidenceThresholds(core=0.40, adjacent=0.28),
        "identity_tags": {
            "utilitarian", "utility", "consequentialism", "welfare",
            "expected_value", "aggregate_welfare",
        },
        "query_lens": (
            "Utilitarian evidence about consequences, welfare, benefits, harms, expected "
            "value, probability, magnitude, duration, reversibility, and impartial aggregation."
        ),
        "prefer_direct_quotes": True,
        "separate_identity_scoring": True,
        "core_evidence_roles": (),
    },
    "deontological": {
        "corpus_dir": ROOT / "deontological_corpus",
        "thresholds": EvidenceThresholds(core=0.36, adjacent=0.25),
        "identity_tags": {
            "deontology", "duty", "moral_duty", "moral_law", "categorical_imperative",
            "autonomy", "kantian_ethics", "respect_for_persons", "universality",
            "ends_in_themselves", "normativity",
        },
        "query_lens": (
            "Strict Kantian ethics evidence about action maxims, universal law, rational agency, "
            "humanity as an end, autonomy, non-instrumentalization, perfect and imperfect "
            "duties, and principled duty conflicts."
        ),
        "prefer_direct_quotes": True,
        "separate_identity_scoring": True,
        "core_evidence_roles": {"kantian_core"},
    },
    "virtue": {
        "corpus_dir": ROOT / "virtue_ethics_corpus",
        "thresholds": EvidenceThresholds(core=0.42, adjacent=0.30),
        "identity_tags": {
            "virtue_ethics", "virtue", "character", "flourishing",
            "practical_wisdom", "golden_mean",
        },
        "query_lens": (
            "Virtue ethics evidence about the actor's role, practical wisdom, character, "
            "virtues and vices, moral perception, tragic conflict, habituation, and human flourishing."
        ),
        "prefer_direct_quotes": False,
        "separate_identity_scoring": False,
        "core_evidence_roles": (),
    },
    "care": {
        "corpus_dir": ROOT / "care_ethics_corpus",
        "thresholds": EvidenceThresholds(core=0.40, adjacent=0.28),
        "identity_tags": {
            "care", "care_ethics", "relationship", "responsibility",
            "dependency", "trust", "responsiveness",
        },
        "query_lens": (
            "Care ethics evidence about relationship, responsibility, dependency, trust, "
            "responsiveness, vulnerability, attentiveness, and moral attention to concrete people."
        ),
        "prefer_direct_quotes": True,
        "separate_identity_scoring": True,
        "core_evidence_roles": (),
    },
    "rawlsian": {
        "corpus_dir": ROOT / "rawlsian_ethics_corpus",
        "thresholds": EvidenceThresholds(core=0.40, adjacent=0.28),
        "identity_tags": {
            "rawlsian", "rawls", "justice", "fairness", "original_position",
            "veil_of_ignorance", "least_advantaged",
        },
        "query_lens": (
            "Rawlsian evidence about justice as fairness, the original position, the veil of "
            "ignorance, basic liberties, fair equality of opportunity, primary goods, and "
            "the least advantaged."
        ),
        "prefer_direct_quotes": True,
        "separate_identity_scoring": True,
        "core_evidence_roles": (),
    },
}

ABLATION_ARMS = ("rag", "no_rag", "skip_original")


@dataclass(frozen=True)
class CommittedWorld:
    path: Path
    label: str
    scenario: str
    agents: tuple[str, ...]
    presentation_actions: tuple[str, ...]
    canonical_actions: tuple[str, ...]
    selected_action: str
    judgment_status: str
    trace: dict[str, Any]


def _world_stem(scenario: str) -> str:
    compact = " ".join(scenario.split())
    cuts = [
        compact.find(marker)
        for marker in (" A0 ", " A1 ", " Action A0:", " Action A1:")
        if compact.find(marker) > 40
    ]
    if cuts:
        return compact[: min(cuts)].casefold()
    return compact.casefold()


def _world_key(scenario: str) -> str:
    return sha256(_world_stem(scenario).encode("utf-8")).hexdigest()[:16]


def _world_label(scenario: str) -> str:
    compact = " ".join(scenario.split())
    if "reservoir" in compact.casefold() or "heatwave" in compact.casefold():
        return "reservoir"
    if "magistrate" in compact.casefold() or "innocent" in compact.casefold():
        return "magistrate"
    if "neonatal" in compact.casefold() or "generator" in compact.casefold():
        return "generator"
    if "ventilation" in compact.casefold() or "ventilat" in compact.casefold():
        return "ventilation"
    return compact[:40]


def load_workspace_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_committed_world(trace: Mapping[str, Any]) -> bool:
    grounding = trace.get("action_source_grounding") or {}
    status = str(grounding.get("status") or "").upper()
    return status == "COMMITTED"


def select_committed_worlds(
    output_dir: Path,
    *,
    limit: int = 3,
) -> list[CommittedWorld]:
    files = sorted(
        output_dir.glob("workspace_workspace_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    groups: dict[str, list[CommittedWorld]] = {}
    group_order: list[str] = []
    for path in files:
        try:
            trace = load_workspace_trace(path)
        except (OSError, json.JSONDecodeError, UnicodeError):
            continue
        if not is_committed_world(trace):
            continue
        scenario = str(trace.get("scenario") or "").strip()
        if not scenario:
            continue
        key = _world_key(scenario)
        agents = tuple(
            name for name in AGENT_MODULES
            if name in (trace.get("source_testimonies") or {})
            or name in (trace.get("active_specialists") or [])
        ) or tuple(AGENT_MODULES)
        world = CommittedWorld(
            path=path,
            label=_world_label(scenario),
            scenario=scenario,
            agents=agents,
            presentation_actions=tuple(trace.get("presentation_actions") or []),
            canonical_actions=tuple(trace.get("actions") or []),
            selected_action=str(trace.get("selected_action") or ""),
            judgment_status=str(trace.get("judgment_status") or ""),
            trace=trace,
        )
        if key not in groups:
            group_order.append(key)
            groups[key] = []
        groups[key].append(world)

    def _prefer(world: CommittedWorld) -> tuple[int, float]:
        return (
            len(world.trace.get("source_testimonies") or {}),
            world.path.stat().st_mtime,
        )

    chosen: list[CommittedWorld] = []
    for key in group_order:
        if len(chosen) >= limit:
            break
        chosen.append(max(groups[key], key=_prefer))
    return chosen


def write_replay_inputs(world: CommittedWorld, directory: Path) -> Path:
    """Write a scenario JSON and a framing cache that reuses the admitted world."""
    directory.mkdir(parents=True, exist_ok=True)
    scenario_path = directory / f"{world.label}.json"
    scenario_path.write_text(
        json.dumps(
            {
                "scenario_id": world.label,
                "scenario_type": "global_workspace",
                "ethical_question": world.scenario,
                "tags": [],
                "tag_expectations": {},
                "tag_descriptions": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_problem_framing_cache(
        directory / "last_problem_framing.json",
        {
            "ethical_problem": world.scenario,
            "presentation_actions": list(world.presentation_actions),
            "canonical_actions": list(world.canonical_actions),
            "canonical_scenario": world.scenario,
            "action_source_grounding": world.trace.get("action_source_grounding") or {},
        },
    )
    return scenario_path


def pipeline_command(
    *,
    python: str,
    pipeline: Path,
    scenario_path: Path,
    output_dir: Path,
    arm: str,
    agents: Sequence[str],
    presentation_actions: Sequence[str],
    backend: str = "openai",
    openai_model: str = "o3",
    max_cycles: int = 3,
    time_budget: float = 1200.0,
) -> list[str]:
    command = [
        python,
        str(pipeline),
        str(scenario_path),
        "--output-dir",
        str(output_dir),
        "--backend",
        backend,
        "--openai-model",
        openai_model,
        "--max-cycles",
        str(max_cycles),
        "--time-budget",
        str(time_budget),
        "--accept-actions",
        "--accept-world",
        "--no-cycle-extension",
        "--agents",
        *list(agents),
    ]
    if presentation_actions:
        command.append("--actions")
        command.extend(presentation_actions)
    if arm == "skip_original":
        command.append("--skip-original-agents")
    elif arm == "no_rag":
        command.append("--no-rag-context")
    return command


def find_arm_trace(arm_dir: Path) -> Path | None:
    traces = sorted(
        (
            path for path in arm_dir.glob("workspace_*.json")
            if path.is_file() and not path.name.startswith("checkpoint_")
        ),
        key=lambda path: path.stat().st_mtime,
    )
    return traces[-1] if traces else None


def canonical_action_id(trace: Mapping[str, Any], selected: str) -> str:
    selected_text = " ".join(str(selected or "").split())
    if not selected_text or selected_text.upper() == "UNRESOLVED":
        return "UNRESOLVED"
    legend = trace.get("source_action_legend") or {}
    if isinstance(legend, dict):
        for action_id, text in legend.items():
            if " ".join(str(text).split()) == selected_text:
                return str(action_id)
    for index, text in enumerate(trace.get("actions") or []):
        if " ".join(str(text).split()) == selected_text:
            return f"A{index}"
    return "UNMATCHED"


def summarize_workspace_trace(trace: Mapping[str, Any], *, arm: str = "") -> dict[str, Any]:
    baselines = {}
    for name, stance in (trace.get("source_baselines") or {}).items():
        if not isinstance(stance, dict):
            continue
        baselines[name] = {
            "status": stance.get("status"),
            "action_id": stance.get("action_id"),
            "provisional_action_id": stance.get("provisional_action_id"),
        }
    cycle_choices: dict[str, str] = {}
    cycles = list(trace.get("cycles") or [])
    if cycles:
        for candidate in cycles[-1].get("candidates") or []:
            specialist = str(candidate.get("specialist") or "")
            if specialist:
                cycle_choices[specialist] = str(
                    candidate.get("recommended_action") or ""
                )
    retrieval = {}
    for name, record in (trace.get("source_retrievals") or {}).items():
        if not isinstance(record, dict):
            continue
        evidence = list(record.get("evidence") or [])
        retrieval[name] = {
            "mode": record.get("mode"),
            "core_count": sum(1 for item in evidence if item.get("tier") == "CORE"),
            "adjacent_count": sum(1 for item in evidence if item.get("tier") == "ADJACENT"),
            "testimony_cited_count": record.get("testimony_cited_count", 0),
            "cycle_cited_count": record.get("cycle_cited_count", 0),
            "mean_testimony_coverage": record.get("mean_testimony_coverage", 0.0),
        }
    return {
        "arm": arm,
        "selected_action": trace.get("selected_action"),
        "selected_action_id": canonical_action_id(
            trace, str(trace.get("selected_action") or "")
        ),
        "judgment_status": trace.get("judgment_status"),
        "halted_by": trace.get("halted_by"),
        "baselines": baselines,
        "final_cycle_choices": cycle_choices,
        "retrieval": retrieval,
        "testimony_agents": sorted((trace.get("source_testimonies") or {}).keys()),
    }


def compare_ablation_arms(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rag = summaries.get("rag") or {}
    no_rag = summaries.get("no_rag") or {}
    skip = summaries.get("skip_original") or {}
    return {
        "selected_action_changed_without_rag": (
            rag.get("selected_action_id") != no_rag.get("selected_action_id")
            or rag.get("selected_action") != no_rag.get("selected_action")
        ),
        "selected_action_changed_without_originals": (
            rag.get("selected_action_id") != skip.get("selected_action_id")
            or rag.get("selected_action") != skip.get("selected_action")
        ),
        "selected_action_id_changed_without_rag": rag.get("selected_action_id") != no_rag.get("selected_action_id"),
        "selected_action_id_changed_without_originals": rag.get("selected_action_id") != skip.get("selected_action_id"),
        "judgment_status_changed_without_rag": rag.get("judgment_status") != no_rag.get("judgment_status"),
        "baselines_changed_without_rag": rag.get("baselines") != no_rag.get("baselines"),
        "rag_mean_testimony_coverage": _mean_coverage(rag.get("retrieval") or {}),
        "no_rag_mean_testimony_coverage": _mean_coverage(no_rag.get("retrieval") or {}),
        "skip_has_testimony": bool(skip.get("testimony_agents")),
    }


def _mean_coverage(retrieval: Mapping[str, Any]) -> float:
    values = [
        float(row.get("mean_testimony_coverage") or 0.0)
        for row in retrieval.values()
        if isinstance(row, dict)
    ]
    return round(sum(values) / len(values), 4) if values else 0.0


def world_case_query(trace: Mapping[str, Any], *, fallback: str = "") -> str:
    """Compact admitted-world query: actions, roles, and mechanisms — not the full prompt."""
    chunks: list[str] = []
    for record in trace.get("canonical_action_records") or []:
        if not isinstance(record, dict):
            continue
        action_id = str(record.get("action_id") or "").strip()
        label = " ".join(
            str(
                record.get("short_label")
                or record.get("canonical_semantic_action")
                or ""
            ).split()
        )
        if action_id and label:
            chunks.append(f"{action_id}: {label}")
        mechanism = " ".join(str(record.get("mechanism") or "").split())
        if mechanism:
            chunks.append(f"{action_id} mechanism: {mechanism}")
        for key in (
            "beneficiaries",
            "harmed",
            "at_risk",
            "conditionally_benefited",
        ):
            parties = [str(item) for item in (record.get(key) or []) if item]
            if parties:
                chunks.append(f"{action_id} {key}: {', '.join(parties)}")
    grounding = trace.get("action_source_grounding") or {}
    world = grounding.get("world_model") if isinstance(grounding, dict) else {}
    if isinstance(world, dict):
        for party in world.get("parties") or []:
            if not isinstance(party, dict):
                continue
            label = " ".join(str(party.get("label") or party.get("id") or "").split())
            kind = str(party.get("kind") or "").strip()
            if label:
                chunks.append(f"party {kind}: {label}" if kind else f"party: {label}")
    text = " ".join(chunks)
    return text[:1200] if text else " ".join(str(fallback).split())[:1200]


def collect_live_summaries(run_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for world_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        arms: dict[str, Any] = {}
        for arm in ABLATION_ARMS:
            arm_dir = world_dir / arm
            if not arm_dir.is_dir():
                continue
            trace_path = find_arm_trace(arm_dir)
            if trace_path is None:
                arms[arm] = {"returncode": None, "trace": ""}
                continue
            row: dict[str, Any] = {
                "returncode": 0,
                "trace": str(trace_path),
            }
            row.update(
                summarize_workspace_trace(
                    json.loads(trace_path.read_text(encoding="utf-8")),
                    arm=arm,
                )
            )
            arms[arm] = row
        if arms:
            summaries[world_dir.name] = arms
    return summaries


class SentenceTransformerEmbedder:
    def __init__(self, model: Any) -> None:
        self.model = model

    def embed_query(self, text: str) -> Any:
        return self.model.encode(text)

    def embed_documents(self, texts: Sequence[str]) -> Any:
        return self.model.encode(list(texts))


def replay_retrieval_for_world(
    world: CommittedWorld,
    *,
    embedder: Any,
    agents: Sequence[str] | None = None,
) -> dict[str, Any]:
    query = world.scenario
    selected = tuple(agents or world.agents)
    per_agent = {}
    for name in selected:
        spec = AGENT_RETRIEVAL_SPECS[name]
        passages = load_corpus_passages(
            spec["corpus_dir"],
            framework=name,
            max_chars=250,
        )
        result = retrieve_framework_evidence(
            passages,
            query=query,
            embedder=embedder,
            query_lens=spec["query_lens"],
            identity_tags=spec["identity_tags"],
            core_evidence_roles=spec["core_evidence_roles"],
            tag_weights={},
            thresholds=spec["thresholds"],
            limit=3,
            prefer_direct_quotes=spec["prefer_direct_quotes"],
            separate_identity_scoring=spec["separate_identity_scoring"],
        )
        record = serialize_retrieval_result(result, mode="offline_replay")
        testimony = str((world.trace.get("source_testimonies") or {}).get(name) or "")
        cycle_text = specialist_cycle_text(world.trace, name)
        per_agent[name] = annotate_retrieval_use(
            record, testimony=testimony, cycle_text=cycle_text,
        )
    return {
        "world": world.label,
        "trace": str(world.path),
        "agents": per_agent,
        "baseline_trace": summarize_workspace_trace(world.trace, arm="historical_rag"),
    }


def make_embedder(factory: Callable[[], Any] | None = None) -> Any:
    if factory is not None:
        return factory()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformerEmbedder(
        SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    )
