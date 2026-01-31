
import json
from pathlib import Path

def format_for_phi4(shard_dir, output_file):
    shard_dir = Path(shard_dir)
    metadata_path = shard_dir / "metadata.jsonl"
    
    formatted_data = []
    
    with open(metadata_path, "r") as f:
        for line in f:
            m = json.loads(line)
            
            # Construct the prompt for Phi-4
            # We want Phi-4 to process the TabPFN latent (injected separately) 
            # and the claim text.
            prompt = f"<|user|>\nClaim: {m['claim']}\nBased on the statistical evidence provided in the latent space, what is the estimated Average Treatment Effect (ATE)?\n<|assistant|>\n<|thought|>\n"
            
            # The target output: 
            # We want the model to perform reasoning (CoT) and then output the ATE.
            # In Phase 3, we'll fine-tune the CoT, but for now let's format the target.
            target = f"The claim states it is a {m['type']}. Based on my analysis, the claim is {'correct' if not m['is_liar'] else 'incorrect'}. The true ATE is {m['ground_truth_ate']:.4f}. <|end|>"
            
            formatted_data.append({
                "id": m["id"],
                "prompt": prompt,
                "target": target,
                "is_liar": m["is_liar"],
                "type": m["type"],
                "ate": m["ground_truth_ate"]
            })
            
    with open(output_file, "w") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    format_for_phi4("data/training_triplets/shard_0", "data/training_triplets/shard_0/phi4_training.jsonl")
    print("Formatted shard_0 for Phi-4 training.")
