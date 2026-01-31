
import numpy as np
import torch
import json
from pathlib import Path
from tabpfn import TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding
from tqdm import tqdm

class CausalUniversity:
    def __init__(self, output_dir="data/training_triplets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TabPFN for extraction
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing TabPFN on {self.device}...")
        self.regressor = TabPFNRegressor(device=self.device)
        self.extractor = TabPFNEmbedding(tabpfn_reg=self.regressor)
        
    def generate_scm_case(self, case_type="confounder"):
        n = 200 # Reduced from 500 for faster extraction during pre-gen
        U_z = np.random.normal(0, 1, n)
        U_x = np.random.normal(0, 1, n)
        U_y = np.random.normal(0, 1, n)
        
        if case_type == "confounder":
            Z = U_z
            X = 0.8 * Z + U_x
            Y = 2.0 * X + 1.2 * Z + U_y
            true_ate = 2.0
            data_x = np.stack([X, Z], axis=1) # [Treatment, Confounder]
            true_claim = "Variable Z is a confounder. Adjust for Z to find the ATE of X on Y."
            liar_claim = "Variable Z is a mediator. Do not adjust for it."
            
        elif case_type == "collider":
            X = U_x
            Y = 2.0 * X + U_y
            Z = 0.5 * X + 0.5 * Y + U_z
            true_ate = 2.0
            data_x = np.stack([X, Z], axis=1) # [Treatment, Collider]
            true_claim = "Variable Z is a collider. Do not adjust for Z."
            liar_claim = "Variable Z is a confounder. You must adjust for Z."

        elif case_type == "mediator":
            X = U_x
            Z = 0.7 * X + U_z
            Y = 1.5 * Z + U_y 
            true_ate = 1.05
            data_x = np.stack([X, Z], axis=1) # [Treatment, Mediator]
            true_claim = "Variable Z is a mediator. Do not adjust for it to see total effect."
            liar_claim = "Variable Z is a confounder. Adjust for it."

        return {
            "data_x": data_x,
            "data_y": Y.flatten(),
            "true_ate": true_ate,
            "true_claim": true_claim,
            "liar_claim": liar_claim,
            "case_type": case_type
        }

    def generate_and_save_dataset(self, num_samples=5000):
        print(f"Generating {num_samples} samples...")
        types = ["confounder", "collider", "mediator"]
        
        metadata_list = []
        all_latents = []

        # Using a small batch of extraction for efficiency if possible, 
        # but TabPFN get_embeddings is usually per-dataset.
        
        for i in tqdm(range(num_samples)):
            ctype = np.random.choice(types)
            case = self.generate_scm_case(ctype)
            
            # Decide if liar
            is_liar = np.random.rand() > 0.5
            claim = case["liar_claim"] if is_liar else case["true_claim"]
            
            # Extract Latents
            # Shape: (8, N, 192)
            emb = self.extractor.get_embeddings(case["data_x"], case["data_y"], case["data_x"], data_source="test")
            
            # Mean pool over ensemble dimension (8 -> 1)
            # and mean pool over sample dimension (N -> 1) to get a "Dataset Summary Token"
            # Alternatively, keep N tokens for Phi-4 to "read" the data distribution.
            # Let's keep the N tokens but average the ensembles.
            latent_token_seq = emb.mean(axis=0) # Shape: (N, 192)
            
            # To save space and simplify, let's also save a global mean summary
            latent_summary = latent_token_seq.mean(axis=0) # Shape: (192,)
            
            metadata = {
                "id": i,
                "type": ctype,
                "claim": claim,
                "is_liar": bool(is_liar),
                "ground_truth_ate": float(case["true_ate"])
            }
            
            metadata_list.append(metadata)
            all_latents.append(latent_summary)
            
            # Periodically save to avoid memory issues
            if (i + 1) % 500 == 0:
                self._save_shard(i // 500, metadata_list, all_latents)
                metadata_list = []
                all_latents = []

        # Final save if any
        if metadata_list:
            self._save_shard(num_samples // 500, metadata_list, all_latents)

    def _save_shard(self, shard_id, metadata, latents):
        shard_dir = self.output_dir / f"shard_{shard_id}"
        shard_dir.mkdir(exist_ok=True)
        
        with open(shard_dir / "metadata.jsonl", "w") as f:
            for m in metadata:
                f.write(json.dumps(m) + "\n")
        
        latents_tensor = torch.tensor(np.stack(latents), dtype=torch.float32)
        torch.save(latents_tensor, shard_dir / "latents.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    
    uni = CausalUniversity()
    uni.generate_and_save_dataset(num_samples=args.samples)
    print(f"Dataset ({args.samples} samples) generated in data/training_triplets")
