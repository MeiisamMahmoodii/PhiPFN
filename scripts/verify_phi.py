
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

model_id = "microsoft/Phi-4-mini-instruct"

print(f"Verifying model: {model_id}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Try loading in 4-bit to check compatibility with bitsandbytes
    # If not on GPU, this might fail or fallback.
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
    ) if torch.cuda.is_available() else None

    # On CPU (Mac), we can't use bitsandbytes 4-bit easily, 
    # but we can check if the model is reachable and loadable in float32/fp16
    print("Loading model metadata...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=compute_dtype
    )
    
    print(f"SUCCESS: Model {model_id} loaded.")
    print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

except Exception as e:
    print(f"FAILED: {e}")
