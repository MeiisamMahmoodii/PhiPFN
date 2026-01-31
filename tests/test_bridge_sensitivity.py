
import torch
import numpy as np
from tabpfn import TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding
from models.bridge import LinearBridge
import torch.nn.functional as F

def test_bridge_sensitivity():
    print("--- Bridge Sensitivity Test ---")
    
    # 1. Generate Datasets
    N = 100
    
    # Dataset A: X -> Y (Linear Cause)
    # X causes Y
    Xa = np.random.normal(0, 1, size=(N, 1))
    Ya = 2.0 * Xa[:, 0] + np.random.normal(0, 0.1, size=N)
    Data_A_X = Xa
    Data_A_Y = Ya
    
    # Dataset B: X | Y (Independent)
    # X and Y are unrelated
    Xb = np.random.normal(0, 1, size=(N, 1))
    Yb = np.random.normal(0, 1, size=N)
    Data_B_X = Xb
    Data_B_Y = Yb
    
    print("Datasets generated.")
    
    # 2. Initialize Models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    regressor = TabPFNRegressor(device=device)
    extractor = TabPFNEmbedding(tabpfn_reg=regressor)
    
    # Initialize Bridge (Randomly initialized)
    bridge = LinearBridge(input_dim=192, output_dim=3072).to(device)
    
    # 3. Extract Latents
    print("Extracting TabPFN embeddings...")
    # shape: (Ensembles, Samples, 192)
    latents_a = extractor.get_embeddings(Data_A_X, Data_A_Y, Data_A_X, data_source="test")
    latents_b = extractor.get_embeddings(Data_B_X, Data_B_Y, Data_B_X, data_source="test")
    
    # Convert to torch
    latents_a = torch.tensor(latents_a, dtype=torch.float32).to(device)
    latents_b = torch.tensor(latents_b, dtype=torch.float32).to(device)
    
    print(f"Latents Shape A: {latents_a.shape}")
    
    # 4. Project through Bridge
    # The bridge does mean pooling over ensemble dim 0
    proj_a = bridge(latents_a) # -> (Samples, 3072)
    proj_b = bridge(latents_b) # -> (Samples, 3072)
    
    print(f"Projected Shape: {proj_a.shape}")
    
    # 5. Measure Similarity
    # We want to compare the "Whole Dataset Representation". 
    # Let's take the mean representation of the dataset to compare "System State A" vs "System State B"
    # Or, typically, we treat them as sequence tokens. 
    # Let's compare the mean vectors of the two datasets.
    
    vec_a = proj_a.mean(dim=0).unsqueeze(0) # (1, 3072)
    vec_b = proj_b.mean(dim=0).unsqueeze(0) # (1, 3072)
    
    similarity = F.cosine_similarity(vec_a, vec_b).item()
    
    print(f"\nCosine Similarity between Causal vs Independent Datasets: {similarity:.4f}")
    
    if similarity > 0.99:
        print("WARNING: High similarity! The bridge might be collapsing information.")
    else:
        print("SUCCESS: Distinct representations detected.")
        print("The random projection preserved the difference between Causal and Independent data.")

if __name__ == "__main__":
    test_bridge_sensitivity()
