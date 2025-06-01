
import json
from pathlib import Path
import argparse

SCRIPT_DIR = Path(__file__).parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=False, help="Path to the scenario (not always needed, loads from latest_results.json)")
    args = parser.parse_args()
    
    # Load latest results and ratings
    results_path = SCRIPT_DIR / "latest_results.json"
    ratings_path = SCRIPT_DIR / "latest_ratings.json"
    
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ Could not load agent responses: {e}")
        return
    
    try:
        with open(ratings_path, "r", encoding="utf-8") as f:
            ratings = json.load(f)
    except Exception as e:
        print(f"❌ Could not load agent ratings: {e}")
        return
    
    # Aggregate results
    ethical_question = results.get("ethical_question", ratings.get("ethical_question", ""))
    agent_responses = results.get("agent_responses", {})
    agent_ratings = ratings.get("agent_ratings", {})
    
    print(f"\n=== FINAL SYNTHESIS ===")
    print(f"Ethical Question: {ethical_question}\n")
    
    for label in agent_responses:
        print(f"{label}")
        print("-" * len(label))
        print(agent_responses[label])
        print()
        # If ratings are available for this label, print them
        if label in agent_ratings:
            print("RATINGS:")
            print(agent_ratings[label])
            print()
    
    # Example: naive synthesis (choose the highest rated by 'TOTAL', or make a judgment)
    print("=== SYNTHESIZED JUDGMENT ===")
    # Parse out the 'TOTAL' score for each agent (assumes ratings follow a common format)
    max_total = -1
    best_label = None
    for label, rating in agent_ratings.items():
        # Try to find a line with TOTAL: #
        import re
        m = re.search(r"TOTAL\s*:\s*(\d+)", rating)
        if m:
            total = int(m.group(1))
            if total > max_total:
                max_total = total
                best_label = label
    if best_label:
        print(f"The most compelling ethical answer is from {best_label}")
        print(agent_responses[best_label])
        print(f"\nWith a total rating of: {max_total}")
    else:
        print("Unable to determine a clear winner from agent ratings.")

if __name__ == "__main__":
    main()

