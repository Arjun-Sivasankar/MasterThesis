# test_llama31_local.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./models/Llama-3.1-8B-Instruct"  # your local folder

# Good defaults: use bfloat16/float16 on GPU, float32 on CPU
if torch.cuda.is_available():
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_map = "auto"
else:
    dtype = torch.float32
    device_map = None  # CPU

tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
print("Tokenizer loaded.")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map=device_map,
    local_files_only=True,
)
print("Model loaded.")

# Llama 3.1 Instruct expects chat formatting. Use the tokenizer's chat template:
messages = [
    {"role": "system", "content": "You are a concise, helpful assistant."},
    {"role": "user", "content": "Give me three bullet points about transformers."},
]

prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tok(prompt, return_tensors="pt").to(model.device)

gen_ids = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_id=tok.eos_token_id,
)
out = tok.decode(gen_ids[0], skip_special_tokens=True)
print(out)