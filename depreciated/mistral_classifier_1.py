import os
import json
import argparse
import re
import sys
sys.path.insert(0, "/Users/benjaminkowal/Documents/Python/venv/lib/python3.11/site-packages")

def inspect_logprobs(logprobs_list, indices):
    print("\nLogprobs for selected examples:")
    for idx in indices:
        # idx is 1-based
        lp = logprobs_list[idx-1]
        print(f"Example {idx}: {lp}")

MODEL_PATH = "/Users/benjaminkowal/Documents/Python/Python Coding/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DATA_PATH = "/Users/benjaminkowal/Documents/Python/Python Coding/llm_reasoning_pipeline/phase_1_diagnostics/notebooks/Training_Examples.txt"
LABELS = ["DEONTOLOGICAL", "NOT_DEONTOLOGICAL"]
MAX_EXAMPLES = 80
DEBUG = True

# Map file labels ("TRUE"/"NOT") to classification labels
LABEL_MAP = {"TRUE": "DEONTOLOGICAL", "NOT": "NOT_DEONTOLOGICAL"}

from tqdm import tqdm
from llama_cpp import Llama
from sklearn.metrics import accuracy_score

def load_data(path):
    """
    Expect a JSONL, JSON, or text file with JSON records:
      {
        "text": "...",
        "label": "positive"  # or numeric
      }
    """
    data = []
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.jsonl', '.jl']:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    elif ext == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        blocks = re.split(r'\n\s*\n', content)
        for block in blocks:
            m_sent = re.search(r'Sentence:\s*"(.+?)"', block)
            m_ans = re.search(r'Answer:\s*(\w+)', block)
            if m_sent and m_ans:
                data.append({'text': m_sent.group(1), 'label': m_ans.group(1)})
            else:
                raise ValueError(f"Cannot parse block:\n{block}")
    elif ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return data

def build_prompt(text, labels):
    """
    Create a definition and few-shot examples for Kant-style classification.
    labels: list of possible labels, e.g. ['DEONTOLOGICAL','NOT_DEONTOLOGICAL']
    """
    definition = (
        "You're a classifier that picks out Kantian deontological arguments.\n"
        "Definition: Deontological ethics holds that the morality of an action is determined by adherence to universal moral duties and principles, regardless of consequences. "
        "Key elements include the Categorical Imperative—act only on maxims you could will as universal laws—respect for persons as ends in themselves, and evaluating actions by intention and duty. "
        "Focus on duty and moral law, not outcomes.\n\n"
        "Use chain-of-thought reasoning:\n"
        "1. Identify whether the sentence appeals to duty (moral law) or to outcomes.\n"
        "2. Then answer with exactly one word: DEONTOLOGICAL or NOT_DEONTOLOGICAL.\n\n"
    )
    # Saved removed negative example:
    # Example 3: Sentence: "Act with compassion because it produces the greatest happiness." Answer: NOT_DEONTOLOGICAL
    # Saved removed positive example:
    # Example 6: Sentence: "Respecting autonomy is required even if it doesn't produce the best consequences." Answer: DEONTOLOGICAL
    examples = [
        (
            f"Example 1:\n"
            f"Sentence: \"If lying were universalized, trust would be impossible — so we must not lie.\"\n"
            f"Answer: {labels[0]}\n\n"
        ),
        (
            f"Example 2:\n"
            f"Sentence: \"Lying is wrong because it often causes harm.\"\n"
            f"Answer: {labels[1]}\n\n"
        ),
        (
            f"Example 3:\n"
            f"Sentence: \"We must tell the truth as long as it promotes social harmony.\"\n"
            f"Answer: {labels[1]}\n\n"
        ),
        (
            f"Example 4:\n"
            f"Sentence: \"The moral worth of an action lies in its outcomes, so we should maximize utility.\"\n"
            f"Answer: {labels[1]}\n\n"
        ),
        (
            f"Example 5:\n"
            f"Sentence: \"We must keep promises even when doing so conflicts with personal gain.\"\n"
            f"Answer: {labels[0]}\n\n"
        ),
        (
            f"Example 6:\n"
            f"Sentence: \"We must follow the categorical law to treat humanity as an end in itself, regardless of the outcome it produces.\"\n"
            f"Answer: {labels[0]}\n\n"
        ),
    ]
    prompt = definition + "".join(examples) + "Now classify this:\n" + f"Sentence: \"{text}\"\nAnswer:"
    return prompt

def predict(model, prompt, max_tokens=16):
    res = model(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        logprobs=5,
        stop=["\n"],
    )
    text = res['choices'][0]['text'].strip()
    logprobs = res['choices'][0]['logprobs']
    return text, logprobs

def main():
    parser = argparse.ArgumentParser(
        description="Run zero/few-shot classification with quantized Mistral model"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=16,
        help="Max tokens to generate for label"
    )
    args = parser.parse_args()

    labels = LABELS

    model_path = MODEL_PATH
    data_path = DATA_PATH

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        seed=42,
        f16_kv=False,
        use_mlock=True,
        n_threads=8,
        logits_all=True,
    )

    data = load_data(data_path)
    texts = [d['text'] for d in data]
    raw_labels = [d['label'] for d in data]
    # Map file labels ("TRUE"/"NOT") to classification labels
    true_labels = [LABEL_MAP.get(lbl, lbl) for lbl in raw_labels]
    if DEBUG:
        print("Raw labels:", raw_labels)
        print("Mapped true labels:", true_labels)

    if DEBUG:
        print(f"Debug mode ON: limiting to first {MAX_EXAMPLES} example(s)")
        texts = texts[:MAX_EXAMPLES]
        true_labels = true_labels[:MAX_EXAMPLES]

    preds = []
    all_logprobs = []
    for idx, text in enumerate(texts):
        if DEBUG:
            print(f"\n=== Example {idx+1}/{len(texts)} ===")
            print("Text:", text)
        prompt = build_prompt(text, labels)
        if DEBUG:
            print("Prompt:\n", prompt)
        out_text, logprobs = predict(llm, prompt, max_tokens=args.max_tokens)
        if DEBUG:
            print("Raw model output:", out_text)
            print("Logprobs:", logprobs)
        pred = out_text.split()[0]
        pred = re.sub(r'[^A-Z_]', '', pred)
        if DEBUG:
            print("Predicted:", pred, "| True Label:", true_labels[idx])
        preds.append(pred)
        all_logprobs.append(logprobs)

    # Detailed per-example results
    print("\nDetailed Results:")
    for idx, (pred, actual) in enumerate(zip(preds, true_labels), start=1):
        print(f"Example {idx}: Predicted = {pred}, Actual = {actual}")
    acc = accuracy_score(true_labels, preds)
    print(f"Accuracy: {acc * 100:.2f}%")

    target = 0.80
    if acc >= target:
        print(f"\n🎉 Success! Achieved {acc*100:.2f}% >= {target*100:.0f}% target.")
    else:
        print(
            f"\n⚠️ Only {acc*100:.2f}% < {target*100:.0f}% target."
            " Consider tuning prompts or adding key words for forced re-evaluation."
        )
    if DEBUG:
        inspect_logprobs(all_logprobs, [27, 34, 39, 61])

if __name__ == "__main__":
    main()
