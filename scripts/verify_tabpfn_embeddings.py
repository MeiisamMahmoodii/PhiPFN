
import torch
from tabpfn import TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding
import numpy as np

# 1. Create a "Toy" Causal Dataset (X -> Y with Confounder Z)
# Z is the confounder
Z = np.random.normal(size=(100, 1))
# X depends on Z
X = 0.5 * Z + np.random.normal(size=(100, 1))
# Y depends on X and Z
# Fix broadcasting: ensure noise is (100, 1) or flatten X, Z
Y = 2.0 * X + 1.5 * Z + np.random.normal(size=(100, 1))

# Combine into a table
data_x = np.hstack([X, Z]) # Features: [Treatment, Confounder]
data_y = Y.flatten()       # Outcome - Ensure it's 1D

print(f"Data X shape: {data_x.shape}")
print(f"Data Y shape: {data_y.shape}")

# 2. Initialize the Extractor
# We use the Regressor because ATE is a continuous value problem
print("Initializing TabPFNRegressor...")
regressor = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu')
print("Initializing TabPFNEmbedding...")
# Updated initialization based on source code: use tabpfn_reg
extractor = TabPFNEmbedding(tabpfn_reg=regressor)

# 3. Extract the "Causal Latents"
# This gives us the internal 'thought process' of TabPFN about this data
print("Extracting embeddings...")
# Based on source code, get_embeddings needs (X_train, y_train, X, data_source)
# We want embeddings for the whole dataset, so we pass it as X and tell it's "test" (to get inference embeddings)
# or "train" if we want training embeddings. For this verification let's treat it as "test" 
# (using same data for train/test is okay for simple verification of shape)
embeddings = extractor.get_embeddings(data_x, data_y, data_x, data_source="test")

print(f"Shape of Latent Space: {embeddings.shape}") 
# Expected: (Samples, Embedding_Dimension)
