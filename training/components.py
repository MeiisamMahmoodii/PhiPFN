
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from typing import Dict, List, Any
import json
from pathlib import Path

class LinearBridge(nn.Module):
    """
    Alignment layer to project TabPFN embeddings (192) to Phi-4 (3072).
    """
    def __init__(self, input_dim=192, output_dim=3072):
        super().__init__()
        self.bridge = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, x):
        # If input is (Ensembles, Batch, Dim), average over ensembles
        if x.dim() == 3:
            x = x.mean(dim=0)
        return self.bridge(x)

class AxiomCausalModel(nn.Module):
    """
    The 'Centaur' model: TabPFN (Bridge) + Phi-4 Mini (LoRA) + Causal Heads.
    """
    def __init__(self, phi_model, bridge, hidden_dim=3072):
        super().__init__()
        self.phi = phi_model # Phi-4 Mini (already wrapped with LoRA)
        self.bridge = bridge # The 192 -> 3072 projection layer
        
        # Additional Heads for PhD "Twin-Loss" Objective
        # 1. Regression Head for direct ATE prediction
        self.ate_head = nn.Linear(hidden_dim, 1)
        
        # 2. Binary Trust Head (optional if not using tokens)
        self.trust_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, input_ids, attention_mask, latents, labels=None):
        # 1. Project TabPFN latents to Phi-4 space
        # latents: (Batch, 192) -> causal_embeds: (Batch, 1, 3072)
        causal_embeds = self.bridge(latents).unsqueeze(1)
        
        # 2. Get standard text embeddings from Phi-4
        inputs_embeds = self.phi.get_input_embeddings()(input_ids)
        
        # 3. Concatenate: [Causal Token] + [Text Prompt]
        # causal_embeds: (B, 1, D), inputs_embeds: (B, L, D) -> (B, L+1, D)
        full_embeds = torch.cat([causal_embeds, inputs_embeds], dim=1)
        
        # Adjust attention_mask for the extra token
        # mask is (B, L), we need (B, L+1)
        batch_size = attention_mask.shape[0]
        extra_mask = torch.ones((batch_size, 1), device=attention_mask.device)
        full_mask = torch.cat([extra_mask, attention_mask], dim=1)
        
        # Adjust labels for the extra token if provided
        # tokens: [CAUSAL] [T1] [T2] ...
        # labels: [IGNORE] [L1] [L2] ...
        full_labels = None
        if labels is not None:
            extra_labels = torch.full((batch_size, 1), -100, device=labels.device)
            full_labels = torch.cat([extra_labels, labels], dim=1)
            
        # 4. Standard Phi-4 Forward Pass with hidden states
        outputs = self.phi(
            inputs_embeds=full_embeds, 
            attention_mask=full_mask, 
            labels=full_labels,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 5. Extract Heads
        # We use the hidden state of the LAST token for regression/trust
        last_hidden = outputs.hidden_states[-1][:, -1, :] # (Batch, Dim)
        
        ate_pred = self.ate_head(last_hidden).squeeze(-1)
        trust_logit = self.trust_head(last_hidden).squeeze(-1)
        
        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
            "ate_pred": ate_pred,
            "trust_logit": trust_logit,
            "hidden_states": outputs.hidden_states
        }

class ConvergenceMonitor:
    """
    Diagnostic tool to track loss balancing in PhD projects.
    """
    def __init__(self):
        self.stats = []

    def log(self, lm_loss, reg_loss, calib_loss):
        # Prevent division by zero
        # Handle both tensors and scalars
        lm_val = float(lm_loss.detach() if torch.is_tensor(lm_loss) else lm_loss)
        reg_val = float(reg_loss.detach() if torch.is_tensor(reg_loss) else reg_loss)
        cal_val = float(calib_loss.detach() if torch.is_tensor(calib_loss) else calib_loss)
        
        reg_to_lm = reg_val / (lm_val + 1e-6)
        cal_to_lm = cal_val / (lm_val + 1e-6)
        
        entry = {
            "lm_loss": lm_val,
            "reg_loss": reg_val,
            "calib_loss": cal_val,
            "reg_ratio": reg_to_lm,
            "cal_ratio": cal_to_lm
        }
        self.stats.append(entry)
        
        # Alerts
        if reg_to_lm > 10.0:
            print(f"⚠️  WARNING: Regression loss is {reg_to_lm:.1f}x higher than LM loss. Consider lowering alpha.")
        if cal_to_lm > 10.0:
            print(f"⚠️  WARNING: Calibration loss is {cal_to_lm:.1f}x higher than LM loss. Consider lowering gamma.")
        
        return f"R/LM: {reg_to_lm:.3f}, C/LM: {cal_to_lm:.3f}"

class CausalDataCollator:
    """
    Data collator that handles both text tokens and TabPFN latent tensors.
    """
    def __init__(self, tokenizer, latent_dir):
        self.tokenizer = tokenizer
        self.latent_dir = Path(latent_dir)
        self.latents_cache = {} # id -> tensor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_ids = [f["id"] for f in features]
        prompts = [f["prompt"] for f in features]
        targets = [f["target"] for f in features]
        ate_list = [f["ate"] for f in features]
        # Trust label: if NOT liar, then TRUSTED (1.0)
        trust_list = [1.0 if not f["is_liar"] else 0.0 for f in features]
        
        # Tokenize text
        full_texts = [p + t for p, t in zip(prompts, targets)]
        tokenized = self.tokenizer(
            full_texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Load latents
        batch_latents = []
        for sample_id in batch_ids:
            shard_id = sample_id // 500
            if shard_id not in self.latents_cache:
                self.latents_cache[shard_id] = torch.load(self.latent_dir / f"shard_{shard_id}" / "latents.pt")
                
            local_id = sample_id % 500
            batch_latents.append(self.latents_cache[shard_id][local_id])
            
        tokenized["latents"] = torch.stack(batch_latents)
        tokenized["ate_labels"] = torch.tensor(ate_list, dtype=torch.float32)
        tokenized["trust_labels"] = torch.tensor(trust_list, dtype=torch.float32)
        
        # Labels for LM loss (Shifted internally by HF models, but we prepare them)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
