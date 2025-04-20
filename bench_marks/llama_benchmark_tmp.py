import time
import multiprocessing
from llama_cpp import Llama

MODEL_PATH = "./llama-2-7b-chat.Q4_K_M.gguf"
N_CTX = 256  # Shorter context = faster
MAX_TOKENS = 32
PROMPT = "What is the capital of France?"
TEMPERATURE =.6 # my insert

# Loop over different thread counts
for threads in [6, 8]:
    print(f"\n==============================")
    print(f"🧪 Testing with {threads} thread(s)")
    print(f"==============================")

    # Load model with specified number of threads
    start_load = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=threads,
        n_gpu_layers=120  # Force CPU-only for testing
    )
    end_load = time.time()

    print(f"🔧 Model loaded in {end_load - start_load:.2f} sec")
    print(f"🧠 Requested threads: {threads}")
    print(f"🧵 Detected threads in object: {llm.n_threads}")
    print(f"💻 Logical CPU cores available: {multiprocessing.cpu_count()}")

    # Run generation
    print(f"⚡ Running prompt: '{PROMPT}'")
    gen_start = time.time()

    first_token_time = None
    tokens = []

    for i, chunk in enumerate(llm(PROMPT, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stream=True)):
        now = time.time()
        if first_token_time is None:
            first_token_time = now
            print(f"⌛ First token after {first_token_time - gen_start:.2f} sec")
        tokens.append(chunk["choices"][0]["text"])

    gen_end = time.time()

    output = "".join(tokens).strip()
    total_time = gen_end - gen_start
    tps = len(tokens) / total_time if total_time > 0 else 0

    # Report
    print("\n✅ Response:")
    print(output)
    print("\n📈 Stats:")
    print(f"Total tokens: {len(tokens)}")
    print(f"Total time: {total_time:.2f} sec")
    print(f"Tokens/sec: {tps:.2f}")