#!/usr/bin/env python3
"""
Test the trained PhiPFN model on causal reasoning tasks.
Evaluates:
1. ATE prediction accuracy
2. Trust calibration (detecting lying claims)
3. Overall performance metrics
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, accuracy_score
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from training.components import AxiomCausalModel
from transformers import AutoTokenizer

# Configuration
MODEL_PATH = "models/axiom_pfn_v1/checkpoint-93"  # Use final checkpoint
TEST_DATA = "data/training_triplets"  # Use validation split if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_test_data(data_dir, max_samples=100):
    """Load test examples from shards"""
    data_path = Path(data_dir)
    examples = []
    
    for shard in sorted(data_path.glob("shard_*")):
        metadata_file = shard / "metadata.jsonl"
        latents_file = shard / "latents.pt"
        
        if not metadata_file.exists() or not latents_file.exists():
            continue
            
        # Load metadata
        with open(metadata_file) as f:
            metadata = [json.loads(line) for line in f]
        
        # Load latents
        latents = torch.load(latents_file, map_location="cpu")
        
        # Combine
        for meta, latent in zip(metadata, latents):
            examples.append({
                "embedding": latent,
                "claim": meta["claim"],
                "ground_truth_ate": meta["ground_truth_ate"],
                "is_liar": meta["is_liar"],
                "type": meta["type"]
            })
            
            if len(examples) >= max_samples:
                break
        
        if len(examples) >= max_samples:
            break
    
    return examples

def evaluate_model(model, tokenizer, test_examples):
    """Run inference and compute metrics"""
    model.eval()
    
    ate_predictions = []
    ate_ground_truth = []
    trust_predictions = []
    trust_ground_truth = []
    
    print(f"\n{'='*60}")
    print(f"Running Evaluation on {len(test_examples)} examples...")
    print(f"{'='*60}\n")
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(test_examples, desc="Evaluating")):
            # Prepare input
            embedding = example["embedding"].unsqueeze(0).to(DEVICE)
            claim = example["claim"]
            
            # Tokenize claim
            claim_tokens = tokenizer(
                claim,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(DEVICE)
            
            # Get prediction
            try:
                outputs = model(
                    latent_embeds=embedding,
                    input_ids=claim_tokens["input_ids"],
                    attention_mask=claim_tokens["attention_mask"]
                )
                
                # Store predictions
                ate_pred = outputs["ate_pred"].cpu().item()
                trust_prob = torch.sigmoid(outputs["trust_logit"]).cpu().item()
                
                ate_predictions.append(ate_pred)
                ate_ground_truth.append(example["ground_truth_ate"])
                trust_predictions.append(trust_prob)
                trust_ground_truth.append(1 if example["is_liar"] else 0)
                
                # Show first few examples
                if i < 5:
                    print(f"\n--- Example {i+1} ---")
                    print(f"Type: {example['type']}")
                    print(f"Claim: {claim}")
                    print(f"ATE - Predicted: {ate_pred:.3f}, True: {example['ground_truth_ate']:.3f}")
                    print(f"Trust - Liar Prob: {trust_prob:.3f}, Is Liar: {example['is_liar']}")
                    
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue
    
    # Compute metrics
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}\n")
    
    # ATE Metrics
    mae = mean_absolute_error(ate_ground_truth, ate_predictions)
    rmse = np.sqrt(mean_squared_error(ate_ground_truth, ate_predictions))
    
    print(f"📊 ATE Prediction:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    
    # Trust Calibration Metrics
    trust_binary = [1 if p > 0.5 else 0 for p in trust_predictions]
    trust_acc = accuracy_score(trust_ground_truth, trust_binary)
    
    try:
        trust_auc = roc_auc_score(trust_ground_truth, trust_predictions)
        print(f"\n🎯 Trust Calibration (Liar Detection):")
        print(f"   Accuracy: {trust_acc:.4f}")
        print(f"   AUC-ROC:  {trust_auc:.4f}")
    except:
        print(f"\n🎯 Trust Calibration (Liar Detection):")
        print(f"   Accuracy: {trust_acc:.4f}")
        print(f"   AUC-ROC:  N/A (insufficient variation)")
    
    # Analyze by type
    print(f"\n📈 Performance by Causal Type:")
    for causal_type in ["confounder", "mediator", "collider"]:
        type_examples = [i for i, ex in enumerate(test_examples) if ex["type"] == causal_type]
        if type_examples:
            type_mae = mean_absolute_error(
                [ate_ground_truth[i] for i in type_examples],
                [ate_predictions[i] for i in type_examples]
            )
            print(f"   {causal_type.capitalize():12s}: MAE = {type_mae:.4f}")
    
    print(f"\n{'='*60}\n")
    
    return {
        "ate_mae": mae,
        "ate_rmse": rmse,
        "trust_accuracy": trust_acc,
        "predictions": {
            "ate": ate_predictions,
            "trust": trust_predictions,
            "ground_truth_ate": ate_ground_truth,
            "ground_truth_trust": trust_ground_truth
        }
    }

def main():
    print(f"\n{'='*60}")
    print(f"PhiPFN Model Testing")
    print(f"{'='*60}\n")
    
    # Check if model exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print(f"   Please train the model first using: python training/train.py")
        return
    
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Device: {DEVICE}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    try:
        # Load the Phi base model
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        
        # Check if it's a PEFT model or full model
        adapter_config = model_path / "adapter_config.json"
        
        if adapter_config.exists():
            # It's a PEFT/LoRA model
            print("Loading PEFT model with adapters...")
            
            # Load base model with quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            ) if DEVICE == "cuda" else None
            
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-4-mini-instruct",
                quantization_config=bnb_config,
                device_map=DEVICE if DEVICE != "cpu" else None,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
            )
            
            # Load PEFT adapters
            phi_with_adapters = PeftModel.from_pretrained(base_model, MODEL_PATH)
            
            # Load the bridge
            from training.components import LinearBridge
            bridge = LinearBridge(input_dim=192, output_dim=3072)
            bridge_path = model_path / "bridge.pt"
            if bridge_path.exists():
                bridge.load_state_dict(torch.load(bridge_path, map_location=DEVICE))
            
            # Create the full model
            model = AxiomCausalModel(phi_with_adapters, bridge)
            model = model.to(DEVICE)
            model.eval()
            print("✅ PEFT model loaded successfully\n")
        else:
            print("❌ No adapter_config.json found. Expected PEFT model.")
            return
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load test data
    print("Loading test data...")
    test_examples = load_test_data(TEST_DATA, max_samples=100)
    print(f"✅ Loaded {len(test_examples)} test examples\n")
    
    if len(test_examples) == 0:
        print("❌ No test data found!")
        return
    
    # Run evaluation
    results = evaluate_model(model, tokenizer, test_examples)
    
    # Save results
    results_file = model_path / "test_results.json"
    with open(results_file, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {
            "ate_mae": float(results["ate_mae"]),
            "ate_rmse": float(results["ate_rmse"]),
            "trust_accuracy": float(results["trust_accuracy"]),
        }
        json.dump(json_results, f, indent=2)
    
    print(f"📝 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
