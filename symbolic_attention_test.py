import os
import torch
from transformers import AutoTokenizer, AutoModel
from symbolic_attention import SymbolicAttention

def integrate_symbolic_attention(model_path):
    # if model_path is a local directory, load from disk; otherwise treat as HF repo
    load_kwargs = {"local_files_only": True} if os.path.isdir(model_path) else {}

    tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kwargs)
    model     = AutoModel.from_pretrained(model_path, **load_kwargs)

    # dummy input text
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.last_hidden_state  # example usage
    return logits

def test_symbolic_attention_forward():
    # small dims so it’s light on memory
    batch_size, seq_len, hidden_size = 1, 4, 16
    num_symbols, sym_dim = 3, 8

    # instantiate the module
    sym_attn = SymbolicAttention(hidden_size=hidden_size, sym_dim=sym_dim)

    # dummy inputs
    token_embeds = torch.randn(batch_size, seq_len, hidden_size)
    symbolic_embeds = torch.randn(batch_size, num_symbols, sym_dim)

    # forward pass
    out = sym_attn(token_embeds, symbolic_embeds)

    # expectations: same shape as token_embeds
    assert out.shape == (batch_size, seq_len, hidden_size), \
        f"Expected output shape {(batch_size, seq_len, hidden_size)}, got {out.shape}"
    print("✅ SymbolicAttention forward pass OK, output shape:", out.shape)

def test_integration_stub():
    try:
        # Example model_path, replace with actual path or repo id as needed
        model_path = "bert-base-uncased"
        logits = integrate_symbolic_attention(model_path)
        print("✅ integrate_symbolic_attention() ran successfully, logits shape:", logits.shape)
    except Exception as e:
        print("⚠️ integrate_symbolic_attention() failed with:", e)

if __name__ == "__main__":
    test_symbolic_attention_forward()
    test_integration_stub()