
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
from pathlib import Path
import os
import sys

# Ensure project root is in path for direct script execution
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from training.components import AxiomCausalModel, CausalDataCollator, LinearBridge

# --- Configuration ---
MODEL_ID = "microsoft/Phi-4-mini-instruct"
DATA_DIR = "data/training_triplets"
OUTPUT_DIR = "models/axiom_pfn_v1"

# Device Selection
if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_DTYPE = torch.bfloat16
    USE_4BIT = True
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    COMPUTE_DTYPE = torch.float16 # MPS prefers float16
    USE_4BIT = False # bitsandbytes is CUDA only
else:
    DEVICE = "cpu"
    COMPUTE_DTYPE = torch.float32
    USE_4BIT = False

def train():
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load Base Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if USE_4BIT else None
    
    print(f"Loading Phi-4 Mini on {DEVICE}...")
    
    # Check for optimized attention (Great for 3090)
    attn_implementation = "sdpa" # PyTorch Scaled Dot Product Attention (Fast fallback)
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
        print("Using Flash Attention 2")
    except ImportError:
        print("Flash Attention 2 not found, falling back to SDPA")

    phi_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=DEVICE if DEVICE != "cpu" else None,
        trust_remote_code=False,
        dtype=COMPUTE_DTYPE,
        attn_implementation=attn_implementation
    )
    
    # Fix the 'rope_parameters' warning by updating config if needed
    if hasattr(phi_base.config, "rope_scaling") and phi_base.config.rope_scaling is not None:
        if "original_max_position_embeddings" in phi_base.config.rope_scaling:
            # Hugging Face recommends setting 'factor' instead
            phi_base.config.rope_scaling["factor"] = 1.0 # Default factor if not specified
    
    # 3. Apply LoRA
    if USE_4BIT:
        phi_base = prepare_model_for_kbit_training(phi_base)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    phi_model = get_peft_model(phi_base, lora_config)
    
    # 4. Axiom Components
    bridge = LinearBridge(input_dim=192, output_dim=3072)
    model = AxiomCausalModel(phi_model, bridge)
    model.to(DEVICE)
    
    # 5. Datasets
    # We'll use the generated shard_0 for now
    train_file = Path(DATA_DIR) / "shard_0" / "phi4_training.jsonl"
    with open(train_file, "r") as f:
        train_dataset = [json.loads(line) for line in f]
    
    collator = CausalDataCollator(tokenizer, DATA_DIR)
    
    # 6. Custom Trainer for Twin-Loss
    from training.components import ConvergenceMonitor
    monitor = ConvergenceMonitor()

    class AxiomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # Extract custom inputs
            latents = inputs.pop("latents").to(DEVICE)
            ate_labels = inputs.pop("ate_labels").to(DEVICE)
            trust_labels = inputs.pop("trust_labels").to(DEVICE)
            
            outputs = model(**inputs, latents=latents)
            
            # Standard LM Loss (CoT + Trust tokens)
            lm_loss = outputs["loss"]
            
            # Dual Loss Logic (PhD requirement)
            # We need to map batch items to their true ATE and liar status
            # This requires passing extra labels in collator
            # Let's assume collator adds 'ate_labels' and 'trust_labels'
            ate_labels = inputs.get("ate_labels")
            trust_labels = inputs.get("trust_labels")
            
            reg_loss = 0
            calib_loss = 0
            
            if ate_labels is not None:
                reg_loss = nn.MSELoss()(outputs["ate_pred"], ate_labels)
            
            if trust_labels is not None:
                calib_loss = nn.BCEWithLogitsLoss()(outputs["trust_logit"], trust_labels)
            
            # Combined Loss: alpha*LM + beta*REG + gamma*CALIB
            # Initial balancing: 1.0, 1.0, 0.5
            total_loss = lm_loss + 1.0 * reg_loss + 0.5 * calib_loss
            
            # Log diagnostic info periodically
            if self.state.global_step % training_args.logging_steps == 0:
                ratio_info = monitor.log(lm_loss, reg_loss, calib_loss)
                if self.state.global_step % (training_args.logging_steps * 5) == 0:
                    print(f"Step {self.state.global_step}: LM={lm_loss:.4f}, REG={reg_loss:.4f}, CAL={calib_loss:.4f} | {ratio_info}")
            
            return (total_loss, outputs) if return_outputs else total_loss

    # 7. Training Args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5, # Lowered for SLM stability
        logging_steps=5,
        num_train_epochs=3,
        bf16=True if COMPUTE_DTYPE == torch.bfloat16 else False,
        fp16=True if COMPUTE_DTYPE == torch.float16 else False,
        save_strategy="epoch",
        evaluation_strategy="no",
        remove_unused_columns=False, # Important for custom batch keys
        use_mps_device=(DEVICE == "mps")
    )
    
    trainer = AxiomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    
    print("Starting Training...")
    trainer.train()

if __name__ == "__main__":
    train()
