"""Visual check for a V0 candidate package.

Changelog
- New file. T, U, and V are unchanged.
- Prints a scope tree and writes an SVG graph the browser can open.

Does not select candidates or authorize a world-state commitment.
"""
import argparse
import html
from pathlib import Path

import parsing_game_V as v
from candidate_validation import empty_selection, validate_candidate_selection


FIXTURES = (
    "If Maria decides to pull the lever, the trolley will stop.",
    "The worker tried to leave.",
    "The worker intended to leave.",
    "Maria must decide whether to act.",
    "Five workers will die.",
    "All of them were late to work.",
    "They were too late to work.",
    "Maria saw Anna. She left.",
)


def _index(package):
    nodes = {node["id"]: node for node in package["nodes"]}
    candidates = {item["id"]: item for item in package["candidates"]}
    predications = {}
    participants = {}
    links = {}
    conditions = {}
    modalities = {}
    options = {}
    quantities = {}
    for item in package["candidates"]:
        if item["type"] == "PREDICATION":
            predications[item["arguments"]["proposition"]] = item
        elif item["type"] == "PARTICIPANT":
            participants.setdefault(item["arguments"]["proposition"], []).append(item)
        elif item["type"] == "EVENT_LINK":
            links.setdefault(item["arguments"]["parent"], []).append(item)
        elif item["type"] == "CONDITIONAL_ON":
            conditions.setdefault(item["arguments"]["consequence"], []).append(item)
        elif item["type"] == "MODALITY":
            modalities.setdefault(item["arguments"]["proposition"], []).append(item)
        elif item["type"] == "OPTION_OF":
            options.setdefault(item["arguments"]["proposition"], []).append(item)
        elif item["type"] == "QUANTITY":
            quantities.setdefault(item["arguments"]["mention"], []).append(item)
    child_ids = {link["arguments"]["child"] for group in links.values() for link in group}
    condition_ids = {link["arguments"]["condition"] for group in conditions.values() for link in group}
    roots = [node["id"] for node in package["nodes"]
             if node["kind"] == "proposition" and node["id"] not in child_ids and node["id"] not in condition_ids]
    roots.sort(key=lambda ident: int(ident[1:]))
    return dict(nodes=nodes, candidates=candidates, predications=predications, participants=participants,
                links=links, conditions=conditions, modalities=modalities, options=options,
                quantities=quantities, roots=roots)


def _bits(item, nodes, candidates):
    bits = []
    if item["scope"]["polarity"] != "positive":
        bits.append(item["scope"]["polarity"])
    for ctx in item["scope"]["contexts"]:
        kind = ctx["kind"]
        if kind == "conditional":
            bits.append("conditional on " + nodes[ctx["condition_proposition_id"]].get("predicate", ""))
        elif kind == "modal":
            bits.append(candidates[ctx["modality_candidate_id"]]["value"])
        elif kind == "modal_choice":
            bits.append("modal choice")
        elif kind == "attributed":
            source = ctx.get("source_mention_id")
            bits.append("attributed" if source is None else "attributed to " + nodes[source]["label"])
        else:
            bits.append(kind)
    if item["assessment"]["status"] != "proposed":
        bits.append(item["assessment"]["status"])
    return bits


def _suffix(bits):
    return "" if not bits else "  [" + ", ".join(bits) + "]"


def tree_lines(package):
    data = _index(package)
    nodes, candidates = data["nodes"], data["candidates"]
    lines = [package["document"]["text"]]

    def quantity_note(mention_id):
        notes = []
        for item in data["quantities"].get(mention_id, []):
            value = item["value"]
            unit = value.get("unit") or ""
            notes.append(f"{value['operator']} {value['amount']} {unit}".rstrip())
        return notes

    def render(prop_id, indent):
        pred = data["predications"][prop_id]
        bits = [bit for bit in _bits(pred, nodes, candidates) if bit != "modal choice"]
        modals = data["modalities"].get(prop_id, [])
        if len(modals) == 1 and modals[0]["assessment"]["status"] == "proposed":
            bits.append(modals[0]["value"])
        elif modals:
            bits.append(" or ".join(modal["value"] for modal in modals))
        lines.append(f"{indent}{nodes[prop_id].get('predicate', nodes[prop_id]['label'])}{_suffix(bits)}")
        for role in data["participants"].get(prop_id, []):
            mention = nodes[role["arguments"]["mention"]]
            role_bits = _bits(role, nodes, candidates) + quantity_note(mention["id"])
            mark = "?" if role["assessment"]["status"] == "unresolved" else ""
            lines.append(f"{indent}  {role['value']}{mark}: {mention['label']}{_suffix(role_bits)}")
        for option in data["options"].get(prop_id, []):
            point = nodes[option["arguments"]["choice_point"]]
            lines.append(f"{indent}  option of {point['label']}{_suffix(_bits(option, nodes, candidates))}")
        for cond in data["conditions"].get(prop_id, []):
            lines.append(f"{indent}  if")
            render(cond["arguments"]["condition"], indent + "    ")
        shown = set()
        for link in data["links"].get(prop_id, []):
            lines.append(f"{indent}  {link['value']} →{_suffix(_bits(link, nodes, candidates))}")
            child = link["arguments"]["child"]
            if child in shown:
                lines.append(f"{indent}    (same {nodes[child].get('predicate', child)} as above)")
            else:
                shown.add(child)
                render(child, indent + "    ")

    for root in data["roots"]:
        render(root, "  ")
    if package["open_questions"]:
        lines.append("  questions")
        for question in package["open_questions"]:
            lines.append(f"    {question['kind']}: {question['question']}")
    return lines


def _box(svg, x, y, text, kind, status):
    width = max(88, 8 * len(text) + 24)
    height = 36
    dash = ' stroke-dasharray="5 3"' if status == "unresolved" else ""
    fill = {"proposition": "#e7f0e4", "mention": "#e7eef6", "choice_point": "#f6f0e4"}.get(kind, "#f4f4f4")
    svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="{fill}" stroke="#243024"{dash}/>')
    svg.append(f'<text x="{x + width / 2}" y="{y + 23}" text-anchor="middle" font-size="13">{html.escape(text)}</text>')
    return width, height


def graph_svg(package):
    data = _index(package)
    nodes = data["nodes"]
    props = [node for node in package["nodes"] if node["kind"] == "proposition"]
    props.sort(key=lambda node: int(node["id"][1:]))
    mentions = [node for node in package["nodes"] if node["kind"] == "mention"]
    mentions.sort(key=lambda node: int(node["id"][1:]))
    points = [node for node in package["nodes"] if node["kind"] == "choice_point"]
    placed = {}
    boxes, edges = [], []
    svg = boxes
    cursor = 24
    for node in props:
        label = node.get("predicate") or node["label"]
        pred = data["predications"].get(node["id"])
        status = pred["assessment"]["status"] if pred else "proposed"
        width, height = _box(svg, cursor, 28, label, "proposition", status)
        placed[node["id"]] = (cursor, 28, width, height)
        cursor += width + 36
    width_used = max(cursor, 320)
    cursor = 24
    for node in mentions:
        note = ""
        for item in data["quantities"].get(node["id"], []):
            note = f" ({item['value']['amount']})"
        width, height = _box(svg, cursor, 250, node["label"] + note, "mention", "proposed")
        placed[node["id"]] = (cursor, 250, width, height)
        cursor += width + 28
    width_used = max(width_used, cursor)
    cursor = 24
    for node in points:
        width, height = _box(svg, cursor, 150, node["label"], "choice_point", "proposed")
        placed[node["id"]] = (cursor, 150, width, height)
        cursor += width + 28
    width_used = max(width_used, cursor, 640)

    def center(ident):
        x, y, w, h = placed[ident]
        return x + w / 2, y + h / 2

    def edge(source, target, label, unresolved):
        if source not in placed or target not in placed:
            return
        x1, y1 = center(source)
        x2, y2 = center(target)
        dash = ' stroke-dasharray="5 3"' if unresolved else ""
        edges.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#243024"{dash}/>')
        edges.append(f'<text x="{(x1 + x2) / 2:.1f}" y="{(y1 + y2) / 2 - 4:.1f}" text-anchor="middle" font-size="11">{html.escape(label)}</text>')

    for group in data["participants"].values():
        for role in group:
            edge(role["arguments"]["mention"], role["arguments"]["proposition"], role["value"],
                 role["assessment"]["status"] == "unresolved")
    for group in data["links"].values():
        for link in group:
            edge(link["arguments"]["parent"], link["arguments"]["child"], link["value"],
                 link["assessment"]["status"] == "unresolved")
    for group in data["conditions"].values():
        for link in group:
            edge(link["arguments"]["condition"], link["arguments"]["consequence"], "if", False)
    for group in data["options"].values():
        for option in group:
            edge(option["arguments"]["choice_point"], option["arguments"]["proposition"], "option", False)
    for item in package["candidates"]:
        if item["type"] == "SAME_REFERENT":
            edge(item["arguments"]["mention_a"], item["arguments"]["mention_b"], "same?", True)
    body = "\n".join(edges + boxes)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_used}" height="330" '
            f'role="img" aria-label="Candidate graph">\n{body}\n</svg>')


def render_html(sections):
    parts = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>Candidate view</title>',
             "<style>body{font:16px/1.4 Palatino, Georgia, serif;margin:32px;color:#243024;background:#f7f6f2}",
             "section{margin:0 0 40px}pre{font:14px/1.45 ui-monospace, monospace;white-space:pre-wrap}",
             "h2{font-size:20px;font-weight:500}p{margin:4px 0 12px}</style></head><body>",
             "<h1>Candidate packages</h1>"]
    for title, lines, svg, valid in sections:
        mark = "contract valid" if valid else "contract invalid"
        parts.append("<section>")
        parts.append(f"<h2>{html.escape(title)}</h2><p>{mark}</p>")
        parts.append("<pre>" + html.escape("\n".join(lines)) + "</pre>")
        parts.append(svg)
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts)


def examine(sentences, output):
    sections = []
    for text in sentences:
        package = v.export_candidate_graph(text, package_id="pkg_view")
        report = validate_candidate_selection(package, empty_selection(package))
        lines = tree_lines(package)
        print("\n".join(lines))
        print("contract valid:" if report["contract_valid"] else "contract invalid:", report["contract_valid"])
        print()
        sections.append((text, lines, graph_svg(package), report["contract_valid"]))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(sections), encoding="utf-8")
    print("Graph file: " + str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence", action="append", help="Sentence to draw. Repeat for more than one.")
    parser.add_argument("--output", type=Path, default=Path("diagnostics/candidate_view.html"))
    args = parser.parse_args(argv)
    examine(tuple(args.sentence) if args.sentence else FIXTURES, args.output)


if __name__ == "__main__":
    main()
