# symbolic_integration_test.py
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from symbolic_attention import SymbolicAttention

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1) Load GPT-2 small
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2", local_files_only=False)
    model     = GPT2LMHeadModel.from_pretrained("distilgpt2", local_files_only=False)
    model.to(device).eval()
    
    # 2) Instantiate your symbolic‐attention layer
    #    GPT2 hidden size is 768
    hidden_size = model.config.hidden_size
    sym_dim     = 64
    sym_attn    = SymbolicAttention(hidden_size=hidden_size, sym_dim=sym_dim).to(device)
    
    # 3) Simple prompt
    prompt = "The weather today is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids     = inputs.input_ids.to(device)
    attention_mask= inputs.attention_mask.to(device)
    
    # 4) Grab GPT-2 token embeddings directly
    with torch.no_grad():
        token_embeds = model.transformer.wte(input_ids)  # (1, seq_len, 768)
    
    # 5) Create some dummy symbol embeddings
    #    e.g. 3 “symbol slots” of dimension sym_dim
    symbol_embeds = torch.randn(1, 3, sym_dim, device=device)
    
    # 6) Run through your symbolic attention
    enhanced_embeds = sym_attn(token_embeds, symbol_embeds)  # (1, seq_len, 768)
    
    # 7) Feed enhanced embeddings back into GPT-2 to get next‐token logits
    with torch.no_grad():
        outputs = model(inputs_embeds=enhanced_embeds, attention_mask=attention_mask)
        logits  = outputs.logits           # (1, seq_len, vocab_size)
    
    # 8) Pick the very next‐token prediction
    next_token_logits = logits[0, -1, :]
    next_token_id     = torch.argmax(next_token_logits).unsqueeze(0)
    next_token_str    = tokenizer.decode(next_token_id)
    
    print(f"Prompt: {prompt!r}")
    print(f"Predicted next token after symbolic attention: {next_token_str!r}")

    # 9) (Optional) Generate a full continuation of 20 new tokens:
    # continuation_ids = model.generate(
    #     inputs_embeds=enhanced_embeds, 
    #     attention_mask=attention_mask,
    #     max_new_tokens=20,
    #     do_sample=False
    # )
    # print("Full continuation:", tokenizer.decode(continuation_ids[0]))
    
if __name__ == "__main__":
    main()