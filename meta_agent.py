# stdlib and third-party imports
import json
import time
import logging
import os
import gc
import psutil
# from virtue_ethics_agent import respond_to_query as respond_virtue
# from care_ethics_agent_beta import respond_to_query as respond_care
# from deontological_agent import respond_to_query as respond_deon
# from utilitarian_agent_beta import respond_to_query as respond_util
from scenario_builder_general import build_scenario
from pathlib import Path
from datetime import datetime
from shutil import copyfile


# Memory logging utility
def log_memory_state():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss // (1024 ** 2)
    print(f"🧠 Python RAM usage: {mem_mb} MB")

os.environ["PYTORCH_MPS_DEVICE"] = "cpu"
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


# Global model path for all agents
MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"


# Setup logging
logging.basicConfig(level=logging.WARNING)

# Constants
TOKEN_LIMIT = 768  # LLM max token input
RESPONSE_TOKEN_BUDGET = 300
SCENARIO_ID = "auto_parliament_001"  # Use dynamic ID in practice
SCENARIO_FILE_PATH = Path(f"virtue_scenarios/{SCENARIO_ID}.json")
DELAY_BETWEEN_AGENTS = 4  # Seconds between agent calls to avoid memory issues

# 1. Build or load the scenario
scenario_path = Path(f"scenarios/{SCENARIO_ID}.json")
# Prompt the user in the terminal if the scenario file does not exist
if not scenario_path.exists():
    user_question = input("📝 Enter an ethical question for this Parliament scenario: ").strip()
    scenario_data = {
        "scenario_id": SCENARIO_ID,
        "scenario_type": "manual",
        "ethical_question": user_question,
        "tags": [],
        "tag_expectations": {},
        "tag_descriptions": {}
    }
    with open(scenario_path, "w") as f:
        json.dump(scenario_data, f, indent=2)
if scenario_path.exists():
    with open(scenario_path, "r") as f:
        scenario_data = json.load(f)
else:
    scenario_data = build_scenario("A man has been dating someone for a short time and they have HIV but are not contegeous, when should they tell them?", id_override=SCENARIO_ID)

gc.collect()

if not scenario_data.get("ethical_question") or len(scenario_data["ethical_question"].strip()) < 10:
    raise ValueError("The ethical question is missing or too short. Please edit the scenario file or enter a longer question.")

question = scenario_data["ethical_question"]

print("📄 Loaded Scenario Data:\n", json.dumps(scenario_data, indent=2))

agent_dirs = [
    "virtue_scenarios", "care_scenarios", 
    "deontology_scenarios", "utilitarian_scenarios"
]
for d in agent_dirs:
    dst_path = Path(d) / f"{SCENARIO_ID}.json"
    if not dst_path.exists():
        Path(d).mkdir(exist_ok=True)
        copyfile(scenario_path, dst_path)
        print(f"✅ Copied scenario to {dst_path}")

del scenario_data
del scenario_path
gc.collect()

#question = scenario_data["ethical_question"]
#question = question
print("❓ Ethical Question Extracted:\n", question)

# Run only one agent at a time for debugging
import importlib

agents_to_run = {
    "Virtue": lambda *args, **kwargs: importlib.import_module("virtue_ethics_agent").respond_to_query(*args, **kwargs),
    #"Care": lambda *args, **kwargs: importlib.import_module("care_ethics_agent_beta").respond_to_query(*args, **kwargs),
    #"Deontology": lambda *args, **kwargs: importlib.import_module("deontological_agent").respond_to_query(*args, **kwargs),
    #"Utilitarianism": lambda *args, **kwargs: importlib.import_module("utilitarian_agent_beta").respond_to_query(*args, **kwargs),
}

responses = {}
for name, responder in agents_to_run.items():
    try:
        gc.collect()
        log_memory_state()
        logging.debug(f"⏳ Calling {name} agent...")
        print(f"\n📤 Prompt sent to {name} agent:\n{'-' * 60}")
        print(f"Question: {question}")
        print(f"Scenario ID: {SCENARIO_ID}")

        responses[name] = responder(question, SCENARIO_ID, scenario_path=SCENARIO_FILE_PATH)

        gc.collect()
        log_memory_state()
        time.sleep(DELAY_BETWEEN_AGENTS)
        logging.debug(f"✅ {name} agent responded.")
    except Exception as e:
        logging.exception(f"❌ Error in {name} agent:")
        responses[name] = f"[ERROR] {str(e)}"

gc.collect()

# 3. Format the input for synthesis
summarized_prompt = "\n\n".join(
    f"{agent} Perspective:\n{response[:600]}" for agent, response in responses.items()
)

# 4. Generate the final synthesis prompt
final_prompt = f"""
You are the Ethical Parliament Synthesis Agent. Below are the perspectives of four ethical frameworks on the same moral scenario. Your task is to synthesize them into a coherent, thoughtful response that:
- Identifies points of agreement and tension
- Represents each framework fairly
- Offers a unified response or acknowledges irresolvable disagreement

Make your answer clear and accessible. Limit your response to {RESPONSE_TOKEN_BUDGET} tokens.

{summarized_prompt}

Parliament's Unified Response:
"""

# 5. Send to LLM (example using Llama)

try:
    # llm = Llama(
    #     model_path="../llama-2-7b-chat.Q4_K_M.gguf",
    #     n_ctx=768,
    #     n_threads=6,
    #     n_gpu_layers=60,
    #     verbose=False
    # )
    # Deferred to avoid preloading
    from llama_cpp import Llama
    import gc

    print("🧠 Initializing synthesis LLM...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=1536,
        n_threads=6,
        n_gpu_layers=8,
        n_batch=64,
        verbose=False
    )

    try:
        print(f"🛠 LLM config: context length = {llm.n_ctx()}")
        print("\n🧾 Prompt sent to Synthesis LLM:\n", final_prompt)
        logging.debug("⏳ Calling LLM for synthesis...")
        result = llm(final_prompt, max_tokens=RESPONSE_TOKEN_BUDGET, temperature=0.7)
        final_response = result["choices"][0]["text"].strip()
    finally:
        print("🧹 Cleaning up synthesis LLM...")
        del llm
        gc.collect()
        log_memory_state()
        time.sleep(2)
except Exception as e:
    logging.exception("❌ Error during LLM synthesis.")
    final_response = f"[ERROR] {str(e)}"

# 6. Output to file
output_dir = Path("agent_outputs")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"response_parliament_{timestamp}.txt"

with open(output_path, "w") as f:
    f.write(f"Ethical Question: {question}\n\n")
    for agent, resp in responses.items():
        f.write(f"{agent} Response:\n{resp}\n\n")
    f.write("Parliament Synthesis:\n")
    f.write(final_response)

gc.collect()
log_memory_state()

print(f"✅ Parliament response saved to {output_path}")
print("🗣 Final Response:\n", final_response)

gc.collect()
log_memory_state()
