import time
import matplotlib.pyplot as plt
from llama_cpp import Llama

# === CONFIG ===
MODEL_PATH = "./llama-2-7b-chat.Q4_K_M.gguf"
PROMPT = "Explain utilitarian ethics in one sentence."
N_THREADS = 6
N_CTX = 512
MAX_TOKENS = 32
LAYER_RANGE = list(range(80, 165, 10))  # Test from 80 to 160 in steps of 10

# === Store Results ===
layer_values = []
tokens_per_sec = []

# === Benchmark Loop ===
for n_gpu_layers in LAYER_RANGE:
    print(f"\n🧪 Benchmarking with n_gpu_layers = {n_gpu_layers}")
    
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=n_gpu_layers
        )

        start = time.time()
        tokens = []

        for chunk in llm(PROMPT, max_tokens=MAX_TOKENS, stream=True):
            tokens.append(chunk["choices"][0]["text"])
        
        end = time.time()
        duration = end - start
        tps = len(tokens) / duration if duration > 0 else 0

        print(f"✅ {tps:.2f} tokens/sec")
        layer_values.append(n_gpu_layers)
        tokens_per_sec.append(tps)

    except Exception as e:
        print(f"❌ Error at n_gpu_layers={n_gpu_layers}: {e}")
        layer_values.append(n_gpu_layers)
        tokens_per_sec.append(0)

# === Plot Results ===
plt.figure(figsize=(10, 6))
plt.plot(layer_values, tokens_per_sec, marker='o')
plt.title("Tokens/sec vs n_gpu_layers (Metal acceleration)")
plt.xlabel("n_gpu_layers")
plt.ylabel("Tokens/sec")
plt.grid(True)
plt.xticks(LAYER_RANGE)
plt.tight_layout()
plt.show()