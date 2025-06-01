import json
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_FILE = SCRIPT_DIR / "latest_results.json"
RATINGS_FILE = SCRIPT_DIR / "latest_ratings.json"

# --- 1. Load Data ---

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

with open(RATINGS_FILE, "r", encoding="utf-8") as f:
    ratings = json.load(f)

ethical_question = results["ethical_question"]
agent_responses = results["agent_responses"]
agent_ratings = ratings["agent_ratings"]

# --- 2. Define Moral Axes (updated to match ratings axes) ---
import re

def extract_axes(rating_text):
    """
    Extracts the four main axis scores from the rating text and returns as a list of floats.
    """
    axes = []
    axis_labels = [
        "Ethical Clarity",
        "Internal Consistency",
        "Relevance to Scenario Details",
        "Alignment with Ethical Framework"
    ]
    for label in axis_labels:
        match = re.search(rf"{label}:\s*(\d+)", rating_text, re.IGNORECASE)
        if match:
            axes.append(float(match.group(1)))
        else:
            axes.append(0.0)
    return axes

# Define axes based on rating categories
moral_axes = [
    "Ethical Clarity",
    "Internal Consistency",
    "Relevance to Scenario Details",
    "Alignment with Ethical Framework",
]

# --- 3. Build Agent Axes Map from Ratings ---
agent_axes_map = {}
for agent, rating_text in agent_ratings.items():
    agent_axes_map[agent] = extract_axes(rating_text)

# --- 4. Find Top-Rated Agent ---

def extract_total_score(text):
    """Extract the total score (assumed integer after 'TOTAL:') from agent rating text."""
    import re
    match = re.search(r"TOTAL:\s*(\d+)", text)
    return int(match.group(1)) if match else 0

# Find the agent with the highest total score
top_agent = max(agent_ratings.items(), key=lambda x: extract_total_score(x[1]))[0]
top_agent_response = agent_responses.get(top_agent, "")
top_agent_axes = agent_axes_map.get(top_agent, [0.5]*len(moral_axes))

# --- 5. Prudential Meta-Agent Logic (Prototype) ---

def prudential_synthesis(top_agent, agent_responses, agent_axes_map, moral_axes):
    """
    Combine the top agent's response with weighted insight from others.
    For now, simply cite top agent but include most 'different' axis perspective.
    """
    # Find the dissenting agent most different on any axis
    max_diff = 0
    dissent_agent = None
    for agent, axes in agent_axes_map.items():
        if agent == top_agent:
            continue
        diff = sum(abs(a-b) for a, b in zip(axes, top_agent_axes))
        if diff > max_diff:
            max_diff = diff
            dissent_agent = agent

    synthesis = (
        f"The prudential synthesis centers the reasoning of the top-rated agent ({top_agent.replace(' Response:', '')}), "
        "but incorporates an important dissent.\n\n"
        f"Primary recommendation:\n{agent_responses[top_agent]}\n\n"
    )
    if dissent_agent:
        synthesis += (
            f"However, the {dissent_agent.replace(' Response:', '')} agent highlights a crucial alternative perspective:\n"
            f"{agent_responses[dissent_agent]}\n"
            "This suggests that while the top recommendation is strong, prudence would also consider...\n"
        )
    return synthesis

synthesis_text = prudential_synthesis(top_agent, agent_responses, agent_axes_map, moral_axes)

# --- 6. Output Section ---

print("\n=== MORAL GRADIENT MAP ===\n")
# Print axis labels as the four ratings (matching ratings script output)
print(f"{'Agent':<25} | " + " | ".join([f"{axis:<15}" for axis in moral_axes]))
print("-"*80)
for agent, axes in agent_axes_map.items():
    print(f"{agent:<25} | " + " | ".join([f"{x:>6.2f}" for x in axes]))
print("\n=== HIGHER ORDER SYNTHESIS RESPONSE ===\n")
print(synthesis_text)
print("\n=== FORMAL DISSENT SUMMARY ===\n")
for agent, axes in agent_axes_map.items():
    if agent == top_agent:
        continue
    print(f"{agent}:")
    print(f"  Main critique: [To fill in: summarize the main critique or blind spot of this agent]\n")
