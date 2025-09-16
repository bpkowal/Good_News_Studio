import os
from pathlib import Path
import subprocess
import sys
import time
import json
import re
import argparse
import asyncio 
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
import unicodedata

# --- Securely load environment variables ---
load_dotenv()  # reads variables from a .env file if present, otherwise falls back to shell env
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY environment variable not set. "
        "Please add it to your environment or .env file."
    )
# -------------------------------------------

# Instantiate v1 Async client
client = AsyncOpenAI(api_key=openai.api_key)


# === Ethics Parliament Synthesis Pipeline ===
# Runs scenario builder, collects agent responses, rates each, and synthesizes final judgment.


SCRIPT_DIR = Path(__file__).parent

# --- Retro-clean previous latest_results.json (defensive) ---
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def strip_ansi(s: str) -> str:
    if not isinstance(s, str):
        return s
    return ANSI_ESCAPE_RE.sub("", s)

def sanitize_agent_text(text: str) -> str:
    """
    Remove debug/noise lines, code fences, ANSI escapes, and obvious headings.
    Returns the main body text. Idempotent: safe to run multiple times.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Normalize & strip ANSI/control noise early
    text = strip_ansi(text)
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")

    # Drop everything inside fenced code blocks ``` ... ```
    cleaned_no_fence = []
    in_fence = False
    for rawline in text.splitlines():
        line = rawline.rstrip("\n")
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        cleaned_no_fence.append(line)
    text = "\n".join(cleaned_no_fence)

    # Patterns / prefixes to strip
    debug_emoji_prefixes = ("✅","⚠️","🔍","🧠","🔧","🔎","📘","🛰️","🚀","💾","📝","📂","📁","🧪","🔬","📊","📈","📉","🧮","ℹ️","🔄","🔁","🧰","🪛","🔗","⏳","⛳","🪵","🪙","🧷","🧯","🧱","🧲")
    debug_lower_prefixes = (
        "debug", "model returned", "primary model error", "trying fallback",
        "loaded", "retrieved", "sample document", "quote used",
        "rawls outbound prompt preview", "deontology agent sent messages",
        "utilitarian agent sent messages", "care agent sending messages",
        "scenario tags", "expanded tag weights", "scenario tag weights",
        "normalized tags", "tag '", "🧠 expanding tag weights",
        "saved output to", "saved parsed rebuttal", "appended", "running ",
        "agent error", "rating error", "consolidated ratings", "wrote consolidated results",
        "o3 usage", "final synthesis", "saved final synthesis", "skipping cache update",
        "cache hit", "updated", "warning:", "[collector]"
    )

    # If a heading like "X Response:" appears, drop everything up to and including it
    heading_re = re.compile(
        r"^[\s\W\u2700-\u27BF\u1F300-\u1FAFF]*"
        r"(virtue|care|deontolog(?:y|ical)|kantian|utilitarian(?:ism)?|rawlsian)"
        r"(?:\s+ethics)?\s+response\s*[:：-]?\s*$",
        re.IGNORECASE
    )

    cleaned_lines = []
    started = False
    for rawline in text.splitlines():
        line = rawline.strip()

        # Skip obvious heading lines entirely
        if heading_re.match(line):
            started = True
            continue

        # Skip empty or debug-prefixed lines until content starts
        if not started:
            if (not line
                or (line and any(line.startswith(p) for p in debug_emoji_prefixes))
                or any(line.lower().startswith(p) for p in debug_lower_prefixes)
                or line.endswith("Response:")
                or line.endswith("Response :")
                or line.lower().endswith("response")):
                continue
            started = True
            cleaned_lines.append(rawline.rstrip())
            continue

        # After content has started, continue skipping obvious debug/status lines
        if (line and any(line.startswith(p) for p in debug_emoji_prefixes)) or any(line.lower().startswith(p) for p in debug_lower_prefixes):
            continue

        cleaned_lines.append(rawline.rstrip())

    # Collapse multiple blank lines
    out = []
    blank = 0
    for ln in cleaned_lines:
        if ln.strip() == "":
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(ln)
    return "\n".join(out).strip()

# --- Helper functions for robust agent response extraction ---
def build_label_regex(name):
    """
    Build a regex for robustly matching the agent's label in output.
    """
    # Accept e.g. "Virtue Ethics Response:", "Virtue response", "Virtue-Response", etc.
    label_core = name.lower()
    if label_core == "deontology":
        label_core = r"deontolog(?:y|ical)|kantian"
    elif label_core == "utilitarian":
        label_core = r"utilitarian(?:ism)?"
    elif label_core == "virtue":
        label_core = r"virtue(?: ethics)?"
    elif label_core == "care":
        label_core = r"care(?: ethics)?"
    elif label_core == "rawlsian":
        label_core = r"rawlsian"
    else:
        label_core = re.escape(name.lower())
    return re.compile(
        r"^[\s\W\u2700-\u27BF\u1F300-\u1FAFF]*"
        r"(" + label_core + r")"
        r"(?:\s+ethics)?\s+response\s*[:：-]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

def extract_agent_response(name, raw):
    """
    Try to robustly extract the agent's response body after its label.
    Returns (body, found_label: bool).
    """
    if not isinstance(raw, str):
        return "", False
    label_re = build_label_regex(name)
    # Find the label line
    lines = raw.splitlines()
    found = False
    out = []
    for i, line in enumerate(lines):
        if label_re.match(line):
            found = True
            # Everything after this line is the response
            out = lines[i+1:]
            break
    if found:
        # Remove leading blank/debug lines
        while out and (not out[0].strip() or out[0].strip().startswith("```")):
            out = out[1:]
        return "\n".join(out).strip(), True
    return raw.strip(), False

print("\n🧹 Retro-cleaning previous latest_results.json (if present)…")
old_results_path = SCRIPT_DIR / "latest_results.json"
if old_results_path.exists():
    try:
        with open(old_results_path, "r", encoding="utf-8") as f:
            raw_json_text = f.read()

        # Backup original before mutating
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = SCRIPT_DIR / f"latest_results_backup_{ts}.json"
        with open(backup_path, "w", encoding="utf-8") as bf:
            bf.write(raw_json_text)

        data = json.loads(raw_json_text)
        if isinstance(data, dict) and isinstance(data.get("agent_responses"), dict):
            for k, v in list(data["agent_responses"].items()):
                if isinstance(v, str):
                    data["agent_responses"][k] = sanitize_agent_text(v)

            # Write cleaned file back
            with open(old_results_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"   ✔ Cleaned {old_results_path.name}; original backed up to {backup_path.name}")
        else:
            print("   ℹ️ latest_results.json did not have the expected shape; left untouched (backup created).")
    except Exception as e:
        print(f"   ⚠️ Retro-clean skipped due to error: {e}")
else:
    print("   ℹ️ No existing latest_results.json to clean.")


# Mapping agent names to their response labels
LABELS = {
    "Utilitarian": "Utilitarian Response:",
    "Virtue": "Virtue Ethics Response:",
    "Deontology": "Deontological Response:",
    "Care": "Care Ethics Response:",
    "Rawlsian": "Rawlsian Ethics Response:"
}

SCENARIO_BUILDER = "scenario_builder_general.py"

AGENTS = [
    ("Virtue", "virtue_ethics_agent_p.py"),
    ("Care", "care_ethics_agent_p.py"),
    ("Deontology", "deontological_agent_p.py"),
    ("Utilitarian", "utilitarian_agent_p.py"),
    ("Rawlsian", "rawlsian_ethics_agent_p.py")
]

SYNTHESIS_RATINGS_SCRIPT = SCRIPT_DIR / "synthesis_ratings_only.py"
SYNTHESIS_SCRIPT = SCRIPT_DIR / "synthesis_final_judgment.py"
PROFILE_PATH = SCRIPT_DIR / "user_ethics_profile.json"

# === 1. Run Scenario Builder ===
print(f"\n🛠 Running Scenario Builder...\n{'='*40}")
try:
    sb_result = subprocess.run(
        ["python", SCENARIO_BUILDER],
        capture_output=True,
        text=True,
        check=True,
    )
    print(sb_result.stdout)
except subprocess.CalledProcessError as e:
    print(f"❌ Scenario Builder Error:\n{e.stderr}")

# === 2. Find the latest scenario file ===
SCENARIO_DIR = Path("scenarios")
scenario_files = sorted(SCENARIO_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
SCENARIO_PATH = str(scenario_files[0]) if scenario_files else ""
if not SCENARIO_PATH:
    print("❌ No scenario file found in 'scenarios/' directory.")
    exit(1)

with open(SCENARIO_PATH, "r", encoding="utf-8") as f:
    scenario_data = json.load(f)

results = {
    "ethical_question": scenario_data.get("ethical_question", ""),
    "agent_responses": {}
}

# === 3. Run Each Agent ===
for name, script in AGENTS:
    print(f"\n🧠 Running {name} Agent...\n{'='*40}")
    try:
        result = subprocess.run(
            ["python", script, "--scenario", SCENARIO_PATH],
            capture_output=True,
            text=True,
            check=True,
        )
        raw = result.stdout
        label = LABELS.get(name, f"{name} Response:")

        # Robust extraction and final sanitation
        clean, found = extract_agent_response(name, raw)
        clean = sanitize_agent_text(clean)
        if not found:
            print(f"[collector] Warning: could not find robust label for '{name}' in stdout; using sanitized full stdout fallback.")
        results["agent_responses"][label] = clean
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} Agent Error:\n{e.stderr}")
    time.sleep(1)  # small pause between agents

# Final defensive sanitization of all agent strings
for k, v in list(results["agent_responses"].items()):
    if isinstance(v, str):
        results["agent_responses"][k] = sanitize_agent_text(v)

# Save consolidated agent responses
output_file = SCRIPT_DIR / "latest_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Wrote consolidated results to {output_file}")

# === 4. Run Ratings Script for Each Agent ===
ratings = {}
for name, label in LABELS.items():
    response = results["agent_responses"].get(label, "")
    if not response:
        continue  # Skip if agent response missing
    try:
        rate_proc = subprocess.run(
            ["python", str(SYNTHESIS_RATINGS_SCRIPT), "--agent-label", label],
            capture_output=True,
            text=True,
            check=True,
        )
        score_text = rate_proc.stdout.strip()
        # Normalise: if the rating script already outputs JSON, parse it;
        # otherwise store the raw string so the final prompt is well‑formed.
        try:
            ratings[label] = json.loads(score_text)
        except json.JSONDecodeError:
            ratings[label] = {"raw_rating": score_text}
    except subprocess.CalledProcessError as e:
        print(f"❌ Rating Error for {label}:\n{e.stderr}")

# Save ratings to JSON
ratings_output = {
    "ethical_question": results["ethical_question"],
    "agent_ratings": ratings
}
ratings_file = SCRIPT_DIR / "latest_ratings.json"
with open(ratings_file, "w", encoding="utf-8") as f:
    json.dump(ratings_output, f, indent=2)
print(f"\n✅ Consolidated ratings:\n{json.dumps(ratings_output, indent=2)}")

# === 6. Run Utilitarian Rebuttal Agent ===
print(f"\n⚖️ Running Utilitarian Rebuttal Agent...\n{'='*40}")
rebuttal = ""
try:
    rebuttal_proc = subprocess.run(
        ["python", "util_rebuttal_agent.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    rebuttal = rebuttal_proc.stdout.strip()
except subprocess.CalledProcessError as e:
    print(f"❌ Rebuttal Agent Error:\n{e.stderr}")


# Save rebuttal to a separate JSON file (read from output text file)
rebuttal_output_file = Path("agent_outputs")  # find most recent rebuttal text file
rebuttal_files = sorted(rebuttal_output_file.glob("util_rebuttal_*.txt"), key=os.path.getmtime, reverse=True)
if rebuttal_files:
    with open(rebuttal_files[0], "r", encoding="utf-8") as f:
        rebuttal_text = f.read().strip()
else:
    rebuttal_text = "[ERROR] No rebuttal output file found."

rebuttal_json = {
    "ethical_question": results["ethical_question"],
    "utilitarian_rebuttal": rebuttal_text
}
rebuttal_file = SCRIPT_DIR / "latest_rebuttal.json"
with open(rebuttal_file, "w", encoding="utf-8") as f:
    json.dump(rebuttal_json, f, indent=2)
print(f"\n✅ Saved parsed rebuttal to {rebuttal_file}")

# === 7. Run Virtue Ethics Rebuttal Agent ===
print(f"\n⚖️ Running Virtue Ethics Rebuttal Agent...\n{'='*40}")
virtue_rebuttal = ""
try:
    virtue_proc = subprocess.run(
        ["python", "virtue_rebuttal_agent.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    virtue_rebuttal = virtue_proc.stdout.strip()
except subprocess.CalledProcessError as e:
    print(f"❌ Virtue Rebuttal Agent Error:\n{e.stderr}")

# Save virtue rebuttal to the same JSON file as utilitarian rebuttal
virtue_output_file = Path("agent_outputs")
virtue_files = sorted(virtue_output_file.glob("virtue_rebuttal_*.txt"), key=os.path.getmtime, reverse=True)
if virtue_files:
    with open(virtue_files[0], "r", encoding="utf-8") as f:
        virtue_text = f.read().strip()
else:
    virtue_text = "[ERROR] No virtue rebuttal output file found."

# Append virtue rebuttal to existing JSON
rebuttal_json["virtue_rebuttal"] = virtue_text
with open(rebuttal_file, "w", encoding="utf-8") as f:
    json.dump(rebuttal_json, f, indent=2)
print(f"\n✅ Appended virtue ethics rebuttal to {rebuttal_file}")

# === 8. Run Deontological Rebuttal Agent ===
print(f"\n⚖️ Running Deontological Rebuttal Agent...\n{'='*40}")
deon_rebuttal = ""
try:
    deon_proc = subprocess.run(
        ["python", "deontology_rebuttal_agent.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    deon_rebuttal = deon_proc.stdout.strip()
except subprocess.CalledProcessError as e:
    print(f"❌ Deontological Rebuttal Agent Error:\n{e.stderr}")

# Save deontological rebuttal to the same JSON file as other rebuttals
deon_output_file = Path("agent_outputs")
deon_files = sorted(deon_output_file.glob("deon_rebuttal_*.txt"), key=os.path.getmtime, reverse=True)
if deon_files:
    with open(deon_files[0], "r", encoding="utf-8") as f:
        deon_text = f.read().strip()
else:
    deon_text = "[ERROR] No deon rebuttal output file found."

# Append deontological rebuttal to existing JSON
rebuttal_json["deontological_rebuttal"] = deon_text
with open(rebuttal_file, "w", encoding="utf-8") as f:
    json.dump(rebuttal_json, f, indent=2)
print(f"\n✅ Appended deontological rebuttal to {rebuttal_file}")

# === 9. Run Care Ethics Rebuttal Agent ===
print(f"\n⚖️ Running Care Ethics Rebuttal Agent...\n{'='*40}")
care_rebuttal = ""
try:
    care_proc = subprocess.run(
        ["python", "care_rebuttal_agent.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    care_rebuttal = care_proc.stdout.strip()
except subprocess.CalledProcessError as e:
    print(f"❌ Care Rebuttal Agent Error:\n{e.stderr}")

# Save care rebuttal to the same JSON file as other rebuttals
care_output_file = Path("agent_outputs")
care_files = sorted(care_output_file.glob("care_rebuttal_*.txt"), key=os.path.getmtime, reverse=True)
if care_files:
    with open(care_files[0], "r", encoding="utf-8") as f:
        care_text = f.read().strip()
else:
    care_text = "[ERROR] No care rebuttal output file found."

# Append care rebuttal to existing JSON
rebuttal_json["care_rebuttal"] = care_text
with open(rebuttal_file, "w", encoding="utf-8") as f:
    json.dump(rebuttal_json, f, indent=2)
print(f"\n✅ Appended care ethics rebuttal to {rebuttal_file}")

# Call o3 for Final synthesis

async def call_o3(messages, model="o3", tool_choice=None, timeout=45):
    """
    Make an async chat completion call using the OpenAI >=1.0.0 interface.
    """
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        tool_choice=tool_choice,  # e.g. {"type": "tool", "name": "python"}
        max_completion_tokens=20048,
        #temperature=0.5,
        timeout=timeout,
    )
    usage = response.usage  # track tokens → cost
    return response.choices[0].message, usage

with open(PROFILE_PATH, "r", encoding="utf-8") as profile_file:
    user_ethics_profile = json.load(profile_file)

# === 10. Build prompt for o3 =================================================
MASTER_PROMPT = """You are the world's foremost expert on negotiation and synthesizing prudent judgments from divergent perspectives. Begin by conducting a concise pre-deliberation (norm-setting) phase: articulate the core values, decision-criteria, and procedural principles that ought to govern the ensuing discussion, drawing on input from all five ethical frameworks (Rawlsian, Care Ethics, Deontological, Utilitarian, and Virtue Ethics). Once these shared norms are sketched, proceed to hear the five ethical Parliament members, each of whom has made initial responses to an ethical question and certain agent's systematic rebuttals to some of their peers' responses. You also have ratings for those responses to consult; weigh them with epistemic humility and decide for yourself how much they matter. Your task is to listen to all arguments, ratings, and rebuttals, identify the strengths of each perspective, and synthesize a more valuable overall recommendation that resolves apparent contradictions. Present your recommendation with epistemic humility, consider the users ethical profile to make your suggestions resonnate with their values, and describe in detail how alternative approaches might also work. After your initial recommendation, simulate the Parliament's comments on your judgment, then review the discussion and outline the final ethical terrain covered by your top recommendation and the next best one. Always take the users current values intod consideration.



The following is an example of the style, depth, and structure to the follow:

*******[Excellent. This is a profound ethical challenge that demands not only a decision but a robust and humble process for arriving at one. As the convener of this Parliament, my role is to synthesize your wisdom, not to impose a pre-ordained conclusion. Let us proceed with clarity, respect, and a shared understanding of the gravity of our task.

\\

Pre-Deliberation: Establishing Norms for a Tragic Choice

Before we can deliberate on the specific case of the student and the teacher, we must agree on the ethical foundation for our discussion. Acknowledging the wisdom inherent in all four of your frameworks—Utilitarian, Deontological, Virtue, and Care—I propose we adopt the following shared principles to govern our process. These are not meant to force consensus on the outcome, but to ensure our path to a recommendation is ethically sound.

\\

1\. Core Value: Respect for Human Dignity and the Promotion of Flourishing

This foundational value acknowledges the Deontological insistence on the inherent, inviolable worth of every individual, while also embracing the Utilitarian, Virtue, and Care ethics' focus on enabling well-being and a flourishing life, both for the individual and their community. Every life has intrinsic value; every decision should also consider its impact on the potential for life to be lived well.

\\

2\. Decision-Criteria: A Multi-faceted Lens

We agree that no single metric can resolve this dilemma. Therefore, our deliberation will weigh four distinct, and sometimes competing, criteria:

\\

A. Maximizing Benefits (The Utilitarian Lens): We will consider the consequences of our decision, impartially assessing which choice is likely to produce the greatest overall good. This includes factors like the number of lives saved and the quality-adjusted life-years preserved.

B. Justice and Fairness (The Deontological Lens): We will assess whether our proposed rule for allocation can be applied universally without contradiction and whether it treats all individuals as ends in themselves, not merely as means to another's end. This criterion demands impartiality and protection of individual rights.

C. Practical Wisdom (The Virtue Ethics Lens): We will consider the character of the decision-making process itself. Is the choice one that a compassionate, just, and wise person would make? This requires moving beyond rigid formulas to embrace the contextual nuances of the situation.

D. Relational Integrity (The Care Ethics Lens): We will examine the decision's impact on the web of relationships surrounding the patients. This includes their families, their communities, and the healthcare providers themselves. The goal is to find a solution that is responsive to and supportive of these vital connections.

3\. Procedural Principles: How We Decide

Our process must be as virtuous as the outcome we hope to achieve. Therefore, we commit to:

\\

Transparency: The reasoning for our final recommendation must be public and clearly articulated.

Consistency: The principles applied here should be applicable to the next similar case.

Humility: We must explicitly acknowledge that we are facing a tragic choice with no perfect solution. Any recommendation is, by nature, an imperfect attempt to find the best path through an impossible situation.

With these norms established, let us now turn to the arguments. I have reviewed your initial responses, your rebuttals, and the associated ratings. The high ratings for the Virtue and Care responses indicate a strong consensus that any purely calculative or rule-based system feels incomplete and risks being inhumane. The rebuttals correctly highlight the core tensions: Deontology’s concern that Virtue Ethics is too subjective, Virtue and Care’s concern that Utilitarianism is too cold, and the Utilitarian concern that Deontology is too rigid to achieve the best outcome.

\\

My task is to find a path forward.

\\

Synthesized Recommendation

Having listened to your distinct and powerful voices, I will not simply choose one framework over the others. Instead, I will synthesize your insights into a recommendation that attempts to honor our shared norms by creating a structured, yet humane, decision-making process.

\\

The apparent contradiction between maximizing good (Utilitarianism) and upholding universal duties (Deontology) can be resolved not by choosing one, but by ordering them within a priority system, guided by the spirit of Virtue and Care ethics.

\\

My recommendation is to allocate the ventilator to the 25-year-old medical student.

\\

However, this conclusion is reached through a specific, multi-tiered ethical framework that must be applied with the utmost compassion and wisdom.

\\

The Multi-Principle Allocation Framework:

\\

Tier 1: Medical Viability and Likelihood of Benefit.

Our first filter must be impartial and evidence-based. We must ask: Who has the greater chance of surviving hospitalization if given the ventilator? In this scenario, the medical student is described as having a "high chance of recovery." While the teacher's prognosis isn't stated, this principle, endorsed by both Utilitarian and Deontological perspectives as a fair starting point, is paramount. For this deliberation, let us assume both patients are deemed likely to benefit, forcing us to a tie-breaker.

\\

Tier 2: The Public Health Tie-Breaker: Prioritizing Instrumental Value.

When two viable patients are in need and resources are critically scarce during a declared public health emergency, the role of the hospital shifts slightly from caring only for the individual to also protecting the health of the entire community. Here, we must lean on the Utilitarian principle of maximizing benefits. Saving the medical student offers a multiplier effect; her survival represents not just one life saved, but the potential for dozens or hundreds of additional lives saved in the future.

\\

This does not mean the teacher's life is of lesser intrinsic worth—a core tenet we must not violate. Rather, it is a recognition that the student's role is of greater instrumental value in the specific context of a pandemic. This is a rule that can be universalized in such emergencies: "In a pandemic, when a choice must be made between two otherwise equal patients, prioritize the one whose survival will most directly contribute to the survival of others."

\\

Tier 3: The Application of Virtue and Care.

This is not a mathematical formula. The decision, even if it follows the Tiers above, must be enacted with the virtues of compassion (Virtue Ethics) and relational responsiveness (Care Ethics).

\\

This means the decision is not made by one person but by a triage committee, removing the crushing burden from the bedside physician and ensuring a more consistent, less arbitrary process.

Communication with the patients and their families must be handled with radical empathy. The reasoning must be explained transparently.

The hospital's duty of care to the teacher does not end. He must be given all other available medical support and comfort care, and his family must be supported through this tragic outcome. His life of contribution must be honored and spoken of with reverence, even in this moment of impossible choice.

Alternative Approach:

\\

The strongest alternative is the strictly Deontological path. After passing the Tier 1 medical viability screen, one could argue the only way to truly respect the equal dignity of both patients is to use an impartial lottery. This approach's greatest strength is that it avoids the "social utility" calculation entirely, refusing to value one life over another for any reason. Its weakness, however, is that it may lead to a clearly worse outcome for the community as a whole, which itself could be seen as an abdication of the duty to protect public health.

\\

Simulated Parliament Comments on the Judgment

Utilitarian Parliament Member: "I concur with the recommendation. Your tiered framework is a pragmatic application of rule utilitarianism. It establishes a clear, consistent policy that, when applied broadly during a crisis, will lead to the best overall outcomes by maximizing life-years and preserving critical healthcare infrastructure. The inclusion of Tier 3 addresses the common critique that our approach lacks compassion, showing that efficiency and humanity are not mutually exclusive."

\\

Deontological Parliament Member: "I must respectfully dissent. While I agree with Tier 1, your Tier 2 violates the categorical imperative. You are explicitly using the retired teacher as a means to an end—the end being the potential future good delivered by the student. The teacher's inherent worth is subordinated to the student's utility. This is an injustice. The moment you introduced 'instrumental value' as a tie-breaker, you created a world where some lives are officially deemed more useful than others. The only just and universalizable tie-breaker is a lottery, which unconditionally respects the equal moral status of both."

\\

Virtue Ethics Parliament Member: "Your synthesis shows admirable phronesis, or practical wisdom. You have grappled with the tragic nature of the choice rather than seeking a simple answer. The emphasis on a committee and compassionate communication in Tier 3 speaks to the importance of the decision-maker's character. However, codifying 'instrumental value' into a rule is concerning. A truly virtuous agent might, in this specific context, still find a compelling reason to choose the teacher—perhaps reflecting the virtue of gratitude for past service or recognizing that a society that discards its elders damages its own moral character in ways a formula cannot capture."

\\

Care Ethics Parliament Member: "You have attempted to integrate our concerns, but the framework remains abstract and hierarchical. It resolves the problem for the hospital administration, but does it truly care for the people involved? A care-based approach would resist a pre-ordained rule and instead demand a process of dialogue, however brief, with the families to understand the relational webs at stake. Your Tier 3 is a humane post-script to a decision already made by a detached rule, rather than an integral part of the decision itself. It does not adequately address the moral injury to the clinicians forced to execute this policy."

\\

Final Ethical Terrain

This deliberation reveals a fundamental and enduring fault line in public ethics. The two most viable paths forward are:

\\

Top Recommendation: A Multi-Principle, Utility-Forward Framework. This approach prioritizes public health resilience. It accepts the difficult but necessary task of making value judgments based on social utility in extreme circumstances, believing this is the most responsible way to maximize lives saved. Its ethical core is a consequentialist-duty to the community, tempered by a procedural commitment to compassion.

\\

Next-Best Recommendation: A Medically-Screened, Egalitarian Lottery. This approach prioritizes absolute individual equality. It refuses to instrumentalize any person, upholding the principle that all lives have equal intrinsic worth, regardless of their age, occupation, or potential. Its ethical core is an unwavering deontological commitment to justice, even if it means forgoing a potentially "better" societal outcome.

\\

Ultimately, the choice between these two frameworks is a choice about what kind of society we wish to be in a crisis: one that organizes itself to save the most people possible, or one that holds the principle of absolute equality sacrosanct, even at a potential cost. My recommendation leans toward the former, but with the profound humility to recognize the moral power and integrity of the latter."] *******
""".strip()

# Helper pretty‑dumpers ---------------------------------------------------
agent_responses_json = json.dumps(results["agent_responses"], indent=2, ensure_ascii=False)
agent_ratings_json = json.dumps(ratings_output["agent_ratings"], indent=2, ensure_ascii=False)
rebuttals_json      = json.dumps(rebuttal_json, indent=2, ensure_ascii=False)
user_profile_json = json.dumps(user_ethics_profile, indent=2, ensure_ascii=False)

# Assemble chat messages --------------------------------------------------
messages = [
    {"role": "system", "content": MASTER_PROMPT},
    {
        "role": "user",
        "content": (
            "### Ethical Question\n"
            f"{results['ethical_question']}"
        )
    },
    {
        "role": "user",
        "content": (
            "### Agent Responses\n"
            "```json\n"
            f"{agent_responses_json}\n"
            "```"
        )
    },
    {
        "role": "user",
        "content": (
            "### Agent Ratings\n"
            "```json\n"
            f"{agent_ratings_json}\n"
            "```"
        )
    },
    {
        "role": "user",
        "content": (
            "### Rebuttals\n"
            "```json\n"
            f"{rebuttals_json}\n"
            "```"
        )
    },
    {
        "role": "user",
        "content": (
            "### User Ethics Profile (Moral Foundations Questionnaire)\n"
            "These values represent moral weightings based on the MFQ (Moral Foundations Questionnaire), scored from 0 to 5. "
            "Higher numbers indicate stronger endorsement.\n\n"
            "```json\n"
            f"{user_profile_json}\n"
            "```"
        )
    },
]


# === 11. Send to o3 and persist synthesis ====================================
async def run_final_synthesis():
    try:
        o3_message, o3_usage = await call_o3(messages, model="o3")
        print("\n🧠 Final Synthesis\n" + "=" * 40)
        print(o3_message.content)

        # Persist final synthesis text
        synthesis_path = SCRIPT_DIR / "latest_synthesis.txt"
        with open(synthesis_path, "w", encoding="utf-8") as f:
            f.write(o3_message.content)
        print(f"\n✅ Saved final synthesis to {synthesis_path}")

        # Log usage metrics
        print(
            f"📝 o3 usage — prompt: {o3_usage.prompt_tokens}, "
            f"completion: {o3_usage.completion_tokens}, "
            f"total: {o3_usage.total_tokens}"
        )
    except Exception as err:
        print(f"❌ o3 Synthesis Error: {err}")

# Kick off the async synthesis run only when executed as a script
if __name__ == "__main__":
    asyncio.run(run_final_synthesis())