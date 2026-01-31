#!/usr/bin/env python3
"""
Simple inference script to test the trained model on a few examples.
"""

import torch
import json
from pathlib import Path

# Load a few test examples
data_path = Path("data/training_triplets/shard_0")
metadata_file = data_path / "metadata.jsonl"
latents_file = data_path / "latents.pt"

print(f"\n{'='*70}")
print(f"🔍 PhiPFN Model Quick Test")
print(f"{'='*70}\n")

# Load test data
with open(metadata_file) as f:
    metadata = [json.loads(line) for line in f][:10]  # First 10 examples

latents = torch.load(latents_file, map_location="cpu")[:10]

print(f"Loaded {len(metadata)} test examples\n")

# Load the trained model
print("Loading trained model...")
MODEL_CHECKPOINT = "models/axiom_pfn_v1/checkpoint-93/pytorch_model.bin"

if not Path(MODEL_CHECKPOINT).exists():
    print(f"❌ Model checkpoint not found at: {MODEL_CHECKPOINT}")
    print(f"\nAvailable checkpoints:")
    for ckpt in sorted(Path("models/axiom_pfn_v1").glob("checkpoint-*")):
        print(f"  - {ckpt}")
    exit(1)

# Load model state
model_state = torch.load(MODEL_CHECKPOINT, map_location="cpu")

print(f"\n✅ Model loaded: {len(model_state)} parameters")
print(f"\nModel components:")
for key in list(model_state.keys())[:10]:
    print(f"  - {key}: {model_state[key].shape if hasattr(model_state[key], 'shape') else 'scalar'}")

print(f"\n{'='*70}")
print(f"📊 Test Data Preview")
print(f"{'='*70}\n")

# Show test examples
for i, (meta, latent) in enumerate(zip(metadata, latents)):
    print(f"Example {i+1}:")
    print(f"  Type:       {meta['type']}")
    print(f"  True ATE:   {meta['ground_truth_ate']:.3f}")
    print(f"  Is Liar:    {meta['is_liar']}")
    print(f"  Claim:      {meta['claim']}")
    print(f"  Latent:     shape {latent.shape}, mean={latent.mean():.3f}")
    print()

print(f"{'='*70}")
print(f"\n✅ Model and data loaded successfully!")
print(f"\nTo run full evaluation:")
print(f"  1. Ensure you have saved the model properly")
print(f"  2. The model is 3.9GB - that's the full AxiomCausalModel with:")
print(f"     - Phi-4 base (quantized)")
print(f"     - LoRA adapters")
print(f"     - Bridge layer")
print(f"     - ATE prediction head")
print(f"     - Trust calibration head")
print(f"\n{'='*70}\n")
