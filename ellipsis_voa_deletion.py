"""Build labeled verb-phrase gaps from VOA Learning English health stories.

A Health & Lifestyle sentence is fully spelled out. This keeps the sentence
as the antecedent, adds a second clause, and deletes the repeated verb phrase.
The deleted words are the gold copy. A row is kept only when the detector
still calls the result a verb-phrase gap and the proposer offers that copy.

Voice of America prose is public domain. Sentences that mention a wire service
are skipped. The dilemma lines and the Hoosier corpus are not used.
"""

import html
import re
import urllib.request

import ellipsis_proposer as proposer

FEED = "https://learningenglish.voanews.com/api/zmmpql-vomx-tpey-_q"
SOURCE = "VOA Learning English, Health & Lifestyle"
WIRE = ("associated press", "reuters", "agence france", "afp")
_MODAL = {"can", "could", "will", "would", "may", "might", "must", "should", "shall"}


def _fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", "replace")


def article_links(feed=FEED, limit=8):
    xml = _fetch(feed)
    links = re.findall(r"<link>(https://learningenglish\.voanews\.com/a/[^<]+)</link>", xml)
    seen = []
    for link in links:
        if link not in seen:
            seen.append(link)
        if len(seen) >= limit:
            break
    return seen


def article_sentences(url):
    page = _fetch(url)
    title = re.search(r"<title>(.*?)</title>", page, re.S)
    title = html.unescape(re.sub("<[^>]+>", " ", title.group(1))).strip() if title else url
    if "words in this story" in page.lower():
        page = re.split(r"Words in This Story", page, maxsplit=1, flags=re.I)[0]
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", page, re.S)
    sentences = []
    for paragraph in paragraphs:
        text = html.unescape(re.sub("<[^>]+>", " ", paragraph))
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 40 or any(mark in text.lower() for mark in WIRE):
            continue
        sentences.append(text)
    return title, sentences


def _remnant_auxiliary(verb):
    """The auxiliary left behind. Aspect and passive stay out of this trial."""
    auxiliaries = [child for child in verb.children if child.dep_ in {"aux", "auxpass"}]
    if any(child.lemma_ in {"be", "have"} for child in auxiliaries):
        return None
    modals = [child for child in auxiliaries if child.lemma_ in _MODAL]
    if modals:
        return modals[-1].text.lower()
    if verb.tag_ == "VBD":
        return "did"
    if verb.tag_ == "VBZ":
        return "does"
    if verb.tag_ in {"VBP", "VB"}:
        return "do"
    return None


def deletions_for_sentence(sentence, nlp):
    """Delete the sentence's main verb phrase, once."""
    if any(mark in sentence for mark in "“”\""):
        return []
    doc = nlp(sentence)
    if len(list(doc.sents)) != 1:
        return []
    verb = next((token for token in doc if token.dep_ == "ROOT" and token.pos_ == "VERB"), None)
    if verb is None or verb.lemma_ in {"do", "say", "tell", "add"}:
        return []
    auxiliary = _remnant_auxiliary(verb)
    ordered, nominals = proposer._phrase(verb)
    if auxiliary is None or not nominals or len(ordered) < 2 or len(ordered) > 12:
        return []
    if verb.tag_ == "VBD":
        auxiliary = "did"
    elif auxiliary not in {"can", "could", "will", "would", "may", "might", "must", "should", "shall"}:
        auxiliary = "does"
    gold = " ".join(token.text for token in ordered)
    lemmas = " ".join(token.lemma_ for token in ordered)
    return [dict(
        gapped=f"{sentence.rstrip()} Sam {auxiliary}, too.",
        gold=gold, lemmas=lemmas, auxiliary=auxiliary,
    )]


def _offered_matches(gold, lemmas, proposals):
    """The offered copy has to be the deleted phrase, not a larger span that contains it."""
    forms = {proposer.normalize(gold), proposer.normalize(lemmas)}
    return any(proposer.normalize(proposal["text"]) in forms for proposal in proposals)


def evaluate_deletion(row, model):
    result = model.propose(row["gapped"])
    proposals = result["proposals"]
    return dict(
        detected=result.get("gate") == "verb_phrase" and bool(proposals),
        recovered=_offered_matches(row["gold"], row["lemmas"], proposals),
        offered=[item["text"] for item in proposals],
        gate=result.get("gate"),
    )


def main():
    nlp = proposer.detector.get_nlp()
    model = proposer.load_proposer()
    attempted = detected = recovered = 0
    shown = 0
    for url in article_links():
        title, sentences = article_sentences(url)
        print(f"\n# {title}")
        for paragraph in sentences:
            for sentence in (span.text.strip() for span in nlp(paragraph).sents):
                for row in deletions_for_sentence(sentence, nlp):
                    attempted += 1
                    outcome = evaluate_deletion(row, model)
                    detected += int(outcome["detected"])
                    recovered += int(outcome["recovered"])
                    if shown < 24:
                        mark = "kept" if outcome["recovered"] else "missed"
                        print(f"  {mark}: {row['gapped']}")
                        print(f"    gold: {row['gold']}")
                        print(f"    offered: {outcome['offered']}")
                        shown += 1
    print(f"\nattempted {attempted}  detected {detected}  recovered {recovered}")


if __name__ == "__main__":
    main()
