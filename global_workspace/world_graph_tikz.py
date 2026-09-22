"""TikZ diagnostics for typed world graphs; JSON remains authoritative."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _tex(value: Any) -> str:
    text = _text(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _node_id(prefix: str, value: Any) -> str:
    return prefix + re.sub(r"[^A-Za-z0-9]+", "", str(value or ""))


def _standalone(title: str, body: str) -> str:
    return "\n".join((
        r"\documentclass[tikz,border=8pt]{standalone}",
        r"\usepackage{xcolor}",
        r"\usetikzlibrary{arrows.meta,positioning,fit,backgrounds}",
        r"\begin{document}",
        rf"% {_tex(title)}",
        body,
        r"\end{document}", "",
    ))


def _party_labels(world: Mapping[str, Any]) -> dict[str, str]:
    return {
        _text(row.get("party_id")): _text(row.get("label"))
        for row in world.get("parties") or [] if isinstance(row, Mapping)
    }


def _effect_style(effect: Mapping[str, Any]) -> str:
    polarity = _text(effect.get("polarity")).upper()
    if polarity == "ADVERSE":
        return "adverse"
    if polarity == "BENEFICIAL":
        return "beneficial"
    if polarity == "FOREGONE":
        return "foregone"
    return "neutral"


def _topological_positions(
    effect_ids: Sequence[str], links: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, int]]:
    ids = list(dict.fromkeys(effect_ids))
    parents = {effect_id: set() for effect_id in ids}
    children = {effect_id: set() for effect_id in ids}
    for link in links:
        source, target = _text(link.get("source_id")), _text(link.get("target_id"))
        if source in parents and target in parents:
            parents[target].add(source)
            children[source].add(target)
    layer = {effect_id: 0 for effect_id in ids}
    queue = [effect_id for effect_id in ids if not parents[effect_id]]
    visited: set[str] = set()
    while queue:
        current = queue.pop(0)
        visited.add(current)
        for child in children[current]:
            layer[child] = max(layer[child], layer[current] + 1)
            if parents[child] <= visited and child not in queue:
                queue.append(child)
    for effect_id in ids:
        if effect_id not in visited:
            layer[effect_id] = max(layer.values(), default=0) + 1
    by_layer: dict[int, list[str]] = {}
    for effect_id in ids:
        by_layer.setdefault(layer[effect_id], []).append(effect_id)
    return {
        effect_id: (depth, index)
        for depth, rows in sorted(by_layer.items())
        for index, effect_id in enumerate(rows)
    }


def render_action_world_tikz(
    world: Mapping[str, Any], action_id: str,
) -> str:
    """Standalone per-action actual-world graph."""
    actions = {
        _text(row.get("action_id")): row
        for row in world.get("actions") or [] if isinstance(row, Mapping)
    }
    effects = [
        row for row in world.get("effects") or []
        if isinstance(row, Mapping) and _text(row.get("action_id")) == action_id
    ]
    effect_ids = {_text(row.get("effect_id")) for row in effects}
    links = [
        row for row in world.get("causal_links") or []
        if isinstance(row, Mapping)
        and _text(row.get("source_id")) in effect_ids
        and _text(row.get("target_id")) in effect_ids
    ]
    positions = _topological_positions([_text(row.get("effect_id")) for row in effects], links)
    parties = _party_labels(world)
    admission = world.get("admission") if isinstance(world.get("admission"), Mapping) else {}
    quarantined = {
        _text(row.get("effect_id")): _text(row.get("contradiction_type"))
        for row in admission.get("quarantined_effects") or []
        if isinstance(row, Mapping) and _text(row.get("effect_id"))
    }
    action = actions.get(action_id, {})
    lines = [
        r"\begin{tikzpicture}[>=Latex,",
        r" event/.style={draw,rounded corners,align=center,text width=3.8cm,minimum height=1.4cm,font=\small},",
        r" beneficial/.style={event,fill=green!12,draw=green!50!black},",
        r" adverse/.style={event,fill=red!10,draw=red!55!black},",
        r" neutral/.style={event,fill=gray!10,draw=gray!65},",
        r" foregone/.style={event,fill=blue!7,draw=blue!45,densely dotted},",
        r" quarantined/.style={event,fill=orange!12,draw=orange!70!black,densely dashed},",
        r" causal/.style={->,thick}, conditional/.style={->,thick,dashed}, quarantineedge/.style={->,thick,densely dashed,draw=orange!70!black}]",
        rf"\node[font=\bfseries,anchor=west] at (0,1.3) {{{_tex(action_id)}: {_tex(action.get('intervention') or action_id)}}};",
    ]
    for effect in effects:
        effect_id = _text(effect.get("effect_id"))
        depth, row = positions.get(effect_id, (0, 0))
        party = parties.get(_text(effect.get("party_id")), _text(effect.get("party_id")))
        modality = _text(effect.get("modality") or "CERTAIN")
        polarity = _text(effect.get("polarity") or "NEUTRAL")
        sources = ", ".join(_text(value) for value in effect.get("clause_ids") or [])
        label = (
            rf"\textbf{{{_tex(effect_id)}}}\\{_tex(effect.get('outcome'))}\\"
            rf"\scriptsize {_tex(party)} · {_tex(polarity)} · {_tex(modality)}"
            + (rf"\\\scriptsize source: {_tex(sources)}" if sources else "")
            + (
                rf"\\\scriptsize QUARANTINED: {_tex(quarantined[effect_id])}"
                if effect_id in quarantined else ""
            )
        )
        lines.append(
            rf"\node[{'quarantined' if effect_id in quarantined else _effect_style(effect)}] ({_node_id('e', effect_id)}) "
            rf"at ({depth * 5.2},{-row * 2.3}) {{{label}}};"
        )
    for link in links:
        source, target = _text(link.get("source_id")), _text(link.get("target_id"))
        relation = _text(link.get("link_relation") or link.get("relation") or "CAUSES")
        condition_ids = [_text(value) for value in link.get("condition_ids") or [] if _text(value)]
        conditional = bool(condition_ids) or _text(link.get("modality")).upper() not in {"", "CERTAIN"}
        style = (
            "quarantineedge" if target in quarantined
            else "conditional" if conditional else "causal"
        )
        if target in quarantined:
            edge_label = f"WITHHELD INFERENCE: {edge_label}"
        edge_label = relation
        if condition_ids:
            edge_label += " if " + ", ".join(condition_ids)
        lines.append(
            rf"\draw[{style}] ({_node_id('e', source)}) -- "
            rf"node[above,font=\scriptsize] {{{_tex(edge_label)}}} ({_node_id('e', target)});"
        )
    lines.append(r"\end{tikzpicture}")
    return _standalone(f"{action_id} actual-world graph", "\n".join(lines))


def render_counterfactual_tikz(world: Mapping[str, Any]) -> str:
    effects = {
        _text(row.get("effect_id")): row
        for row in world.get("effects") or [] if isinstance(row, Mapping)
    }
    parties = _party_labels(world)
    links = [row for row in world.get("counterfactual_links") or [] if isinstance(row, Mapping)]
    endpoint_ids = list(dict.fromkeys(
        value for row in links for value in (
            _text(row.get("source_effect_id")), _text(row.get("alternative_effect_id")),
        ) if value
    ))
    lines = [
        r"\begin{tikzpicture}[>=Latex,event/.style={draw,rounded corners,align=center,text width=4cm,minimum height=1.2cm,font=\small},counter/.style={->,thick,densely dotted,draw=blue!65!black}]",
        r"\node[font=\bfseries,anchor=west] at (0,1.3) {Cross-action counterfactual topology};",
    ]
    for index, effect_id in enumerate(endpoint_ids):
        effect = effects.get(effect_id, {})
        party = parties.get(_text(effect.get("party_id")), _text(effect.get("party_id")))
        label = rf"\textbf{{{_tex(effect_id)}}}\\{_tex(effect.get('outcome'))}\\\scriptsize {_tex(party)}"
        lines.append(rf"\node[event] ({_node_id('e', effect_id)}) at ({(index % 2) * 7},{-(index // 2) * 2.2}) {{{label}}};")
    for row in links:
        source, target = _text(row.get("source_effect_id")), _text(row.get("alternative_effect_id"))
        relation = _text(row.get("counterfactual_relation") or row.get("relation"))
        lines.append(rf"\draw[counter] ({_node_id('e', source)}) -- node[above,font=\scriptsize] {{{_tex(relation)}}} ({_node_id('e', target)});")
    lines.append(r"\end{tikzpicture}")
    return _standalone("Counterfactual topology", "\n".join(lines))


def render_source_binding_tikz(
    world: Mapping[str, Any], clauses: Sequence[Mapping[str, Any]],
) -> str:
    effects = [row for row in world.get("effects") or [] if isinstance(row, Mapping)]
    used_clause_ids = list(dict.fromkeys(
        _text(value) for effect in effects for value in effect.get("clause_ids") or [] if _text(value)
    ))
    clause_by_id = {
        _text(row.get("clause_id")): _text(row.get("text"))
        for row in clauses if isinstance(row, Mapping)
    }
    lines = [
        r"\begin{tikzpicture}[>=Latex,source/.style={draw,rounded corners,align=left,text width=5.3cm,fill=yellow!8,font=\small},event/.style={draw,rounded corners,align=left,text width=5.3cm,fill=gray!7,font=\small},binding/.style={->,thin,draw=blue!55}]",
        r"\node[font=\bfseries] at (0,1.2) {Source facts};",
        r"\node[font=\bfseries] at (8,1.2) {World graph nodes};",
    ]
    for index, clause_id in enumerate(used_clause_ids):
        lines.append(rf"\node[source] ({_node_id('c', clause_id)}) at (0,{-index * 1.8}) {{\textbf{{{_tex(clause_id)}}} {_tex(clause_by_id.get(clause_id, '(action label or unavailable source)'))}}};")
    for index, effect in enumerate(effects):
        effect_id = _text(effect.get("effect_id"))
        lines.append(rf"\node[event] ({_node_id('e', effect_id)}) at (8,{-index * 1.8}) {{\textbf{{{_tex(effect_id)}}} {_tex(effect.get('outcome'))}}};")
        for clause_id in effect.get("clause_ids") or []:
            if _text(clause_id) in used_clause_ids:
                lines.append(rf"\draw[binding] ({_node_id('c', clause_id)}) -- ({_node_id('e', effect_id)});")
    lines.append(r"\end{tikzpicture}")
    return _standalone("Source-to-world bindings", "\n".join(lines))


def write_world_graph_tikz_bundle(
    destination: Path,
    *,
    world_model: Mapping[str, Any],
    clauses: Sequence[Mapping[str, Any]] = (),
    representation_stage: str = "UNSPECIFIED",
) -> dict[str, Any]:
    """Write deterministic `.tex` diagnostics without invoking LaTeX."""
    destination.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    action_ids = [
        _text(row.get("action_id")) for row in world_model.get("actions") or []
        if isinstance(row, Mapping) and _text(row.get("action_id"))
    ]
    for action_id in action_ids:
        path = destination / f"action_{re.sub(r'[^A-Za-z0-9_-]+', '_', action_id)}.tex"
        path.write_text(render_action_world_tikz(world_model, action_id), encoding="utf-8")
        files.append(path.name)
    source_path = destination / "source_bindings.tex"
    source_path.write_text(render_source_binding_tikz(world_model, clauses), encoding="utf-8")
    files.append(source_path.name)
    if world_model.get("counterfactual_links"):
        counter_path = destination / "counterfactual_topology.tex"
        counter_path.write_text(render_counterfactual_tikz(world_model), encoding="utf-8")
        files.append(counter_path.name)
    manifest = {
        "diagnostic_format": "TIKZ_SOURCE_V1",
        "canonical_representation": "JSON_WORLD_MODEL",
        "representation_stage": _text(representation_stage).upper(),
        "latex_compiled": False,
        "files": files,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def compile_world_graph_tikz_bundle(destination: Path) -> dict[str, Any]:
    """Compile a previously written bundle when ``pdflatex`` is available."""
    executable = shutil.which("pdflatex")
    if executable is None:
        return {"status": "LATEX_NOT_AVAILABLE", "compiled": [], "failed": []}
    compiled: list[str] = []
    failed: list[dict[str, Any]] = []
    for source in sorted(destination.glob("*.tex")):
        result = subprocess.run(
            [executable, "-interaction=nonstopmode", "-halt-on-error", source.name],
            cwd=destination,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            compiled.append(source.with_suffix(".pdf").name)
        else:
            failed.append({
                "source": source.name,
                "returncode": result.returncode,
                "stderr_tail": result.stderr[-2000:],
                "stdout_tail": result.stdout[-2000:],
            })
    return {
        "status": "COMPILED" if not failed else "PARTIAL_FAILURE",
        "compiled": compiled,
        "failed": failed,
    }
