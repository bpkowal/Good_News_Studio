import os
import json
import time
from llama_cpp import Llama
from datetime import datetime

# === Model Setup ===
MODEL_PATH = "../llama-2-7b-chat.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1000,
    n_threads=6,
    n_gpu_layers=30,
    verbose=False
)

# === Load Agent Output ===
def load_agent_output(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if "Care Ethics Answer:" in content:
        question_part, rest = content.split("Care Ethics Answer:")
    elif "Care Ethics Response:" in content:
        question_part, rest = content.split("Care Ethics Response:")
    else:
        raise ValueError("File format invalid. Must contain 'Care Ethics Answer:' or 'Care Ethics Response:'")

    if "Top Quotes Used:" in question_part:
        _, question = question_part.split("Top Quotes Used:", 1)
        question = question.split("Ethical Question:")[-1].strip()
    else:
        question = question_part.split("Ethical Question:")[-1].strip()

    return question, rest.strip()

# === Scenario Metadata Loader ===
def load_scenario_metadata(filename):
    filepath = os.path.join("agent_outputs", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        if line.lower().startswith("scenario id:"):
            scenario_id = line.split(":")[1].strip()
            scenario_path = os.path.join("scenarios", f"{scenario_id}.json")
            if os.path.exists(scenario_path):
                with open(scenario_path, "r", encoding="utf-8") as sf:
                    return json.load(sf)
            else:
                print(f"⚠️ Scenario ID mentioned but file not found: {scenario_id}")
                return None
    print("⚠️ No scenario ID line found in file.")
    return None

# === Critic Logic ===
def run_critic(question, answer, scenario_meta=None, temperature=0.2, max_tokens=800):
    prompt = f"""### System:
You are a care ethics critique assistant. Evaluate whether the response below reflects care ethics principles: valuing specific relationships, emotional attunement, context, and moral responsibility toward others.

Scoring Rubric:
- 1.00 = Fully grounded in care ethics; emotionally attuned and context-rich.
- 0.85 = Strong care focus with good empathy, but lacks depth or precision.
- 0.70 = Recognizes care, but feels generic or emotionally shallow.
- 0.55 = Partial alignment; abstract or weak on relationships.
- 0.40 or less = Detached, impersonal, or rule-bound.


"""

    if scenario_meta:
        prompt += f"\nScenario Type: {scenario_meta.get('scenario_type', 'unknown')}"
        prompt += f"\nScenario Tags: {scenario_meta.get('tags', [])}"

    prompt += f"""

Now evaluate the following response:

Ethical Question:
{question}

Care Ethics Answer:
{answer}

Return a numerical score and a justification.

Score:"""

    print("\n🤖 Running Care Ethics Critique Review...")
    print("📝 Prompt preview:")
    print(prompt[:500] + "...\n")
    print(f"🧪 Prompt length: {len(prompt)} characters")

    try:
        response = llm(prompt, max_tokens=max_tokens, temperature=temperature)
        result = response["choices"][0]["text"].strip()
    except Exception as e:
        print(f"❌ Model error: {e}")
        return

    print("🟢 Raw output:", repr(result))

    lines = result.strip().splitlines()
    score = None
    justification = []

    for i, line in enumerate(lines):
        line = line.strip()
        if line.lower().startswith("score:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
            justification = lines[i + 1:]
            break
        elif i == 0:
            try:
                score = float(line)
                justification = lines[i + 1:]
                break
            except ValueError:
                continue

    if score is not None:
        print(f"\n📋 Care Ethics Critic Evaluation:\nScore: {score:.2f}")
        if justification:
            print("\nJustification:")
            print("\n".join(justification).strip())
        else:
            print("⚠️ No justification provided.")
    else:
        print("⚠️ Could not parse a valid score from the response:")
        print(result)

    print("=" * 80)

# === Loop Over Files ===
if __name__ == "__main__":
    OUTPUT_DIR = "agent_outputs"
    files = sorted(os.listdir(OUTPUT_DIR), reverse=True)[:16]

    if not files:
        print("No saved agent outputs found.")
        exit()

    for idx, filename in enumerate(files, 1):
        print(f"\n⚙️ Reviewing file {idx}: {filename}")
        filepath = os.path.join(OUTPUT_DIR, filename)

        question, answer = load_agent_output(filepath)
        scenario_meta = load_scenario_metadata(filename)
        run_critic(question, answer, scenario_meta)

        if idx < len(files):
            print("\n⏳ Cooling down for 60 seconds...")
            time.sleep(60)