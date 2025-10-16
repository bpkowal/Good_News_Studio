import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import heapq

try:
    import yaml
except ImportError:
    yaml = None

# Simple relevance scorer (term frequency + presence bonus)
def _score(query: str, text: str) -> float:
    q_terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    t = text.lower()
    return sum(t.count(term) * 2 + (1 if term in t else 0) for term in q_terms)

def _parse_frontmatter_md(md_text: str) -> (Optional[Dict[str, Any]], str):
    """
    If md_text starts with YAML frontmatter, parse it and return (meta, body).
    Otherwise return (None, full_text).
    """
    if not md_text.startswith("---\n"):
        return None, md_text
    parts = md_text.split("---", 2)
    if len(parts) < 3:
        return None, md_text
    raw_meta = parts[1]
    body = parts[2].lstrip("\n")
    if yaml is None:
        return {"_raw": raw_meta.strip()}, body
    try:
        meta = yaml.safe_load(raw_meta) or {}
    except Exception:
        return None, md_text
    return meta, body

def _read_quotes_md(path: Path) -> List[Dict[str, Any]]:
    """
    Reads a combined markdown file for a school.
    Returns a list of dicts: {"title": ..., "quote": ..., "source": "", "meta": {...}}.
    It extracts sections starting with ## header, followed by description and blockquote lines.
    """
    text = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter_md(text)

    pattern = re.compile(
        r"^##\s*(.+?)\s*$"          # header line
        r"((?:\n(?!##).*)*)"        # description lines (not starting with ##)
        r"((?:\n>.*(?:\n>.*)*)+)",  # one or more blockquote lines
        re.MULTILINE
    )

    quotes = []
    for match in pattern.finditer(body):
        title = match.group(1).strip()
        desc_block = match.group(2).strip()
        quote_block = match.group(3).strip()

        # Remove any '---' lines in desc_block and quote_block
        desc_lines = [line for line in desc_block.splitlines() if line.strip() != "---"]
        quote_lines = [line for line in quote_block.splitlines() if line.strip() != "---"]

        # Combine description and quote block
        combined_quote = "\n".join(desc_lines + quote_lines)

        quotes.append({
            "title": title,
            "quote": combined_quote,
            "source": "",
            "meta": meta
        })

    return quotes

def shortlist_quotes(school: str, scenario: str, root="quotes_combined", k_local=20) -> List[Dict[str, Any]]:
    """
    Reads the `school.md` file under `quotes_combined/`, scores its chunks, and returns
    the top k_local quotes.
    """
    path = Path(root) / f"{school}.md"
    if not path.exists():
        return []
    quotes = _read_quotes_md(path)
    scored = [(-_score(scenario, q["quote"]), idx, q) for idx, q in enumerate(quotes)]
    top = heapq.nsmallest(k_local, scored)
    return [q for _, _, q in top]

def pick_top3_with_llm(scenario: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    provider = os.getenv("EP_SELECTOR_PROVIDER", "openai")
    model = os.getenv("EP_SELECTOR_MODEL", "gpt-4o-mini")

    lines = []
    for i, q in enumerate(candidates, 1):
        lines.append(f"{i}. {q['quote']}  — {q.get('source','')}")

    prompt = (
        "Select the BEST 3 quotes (by number) that help analyze the scenario. "
        "Return ONLY a JSON list of the chosen numbers.\n\n"
        f"SCENARIO:\n{scenario}\n\nCANDIDATES:\n" + "\n".join(lines)
    )

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        msg = [
            {"role":"system","content":"You are a precise selector. Output JSON only."},
            {"role":"user","content": prompt}
        ]
        resp = client.chat.completions.create(model=model, messages=msg, temperature=0)
        content = resp.choices[0].message.content.strip()
    else:
        import requests, json as pyjson
        hf_url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY','')}"}
        payload = {"inputs": prompt, "parameters": {"temperature": 0}}
        r = requests.post(hf_url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()[0].get("generated_text", "")

    import json as pyjson
    try:
        nums = pyjson.loads(content)
        idxs = [int(n) for n in nums][:3]
    except Exception:
        idxs = [1,2,3]

    chosen = []
    for i in idxs:
        if 1 <= i <= len(candidates):
            chosen.append(candidates[i-1])
    return chosen

def select_quotes_for_agent(school: str, scenario: str, k_local=20) -> List[Dict[str, Any]]:
    candidates = shortlist_quotes(school, scenario, k_local=k_local)
    return pick_top3_with_llm(scenario, candidates)