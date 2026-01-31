
import numpy as np
import torch
from tabpfn import TabPFNClassifier
import torch.nn as nn

def test_latent_extraction():
    print("Initializing TabPFN...")
    # Initialize with default settings
    classifier = TabPFNClassifier(device='cpu', n_estimators=1)
    
    # We need to fit it once to initialize the underlying model if it's lazy loaded
    # Although for TabPFN v2, the model might be loaded on init or fit.
    # The verify script showed model_ attribute exists after fit.
    
    X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    y = np.array([1, 1, 0, 0])
    
    print("Fitting to ensure model is initialized...")
    classifier.fit(X, y)
    
    # Access the underlying PyTorch model
    # Based on previous inspection, it should be in classifier.models_[0] or classifier.model_
    if hasattr(classifier, 'model_'):
        model = classifier.model_
    elif hasattr(classifier, 'models_'):
        model = classifier.models_[0]
    else:
        raise ValueError("Could not find underlying PyTorch model")
        
    print(f"Model found: {type(model)}")
    
    # Define a hook to capture the output of the transformer_encoder
    # The inspection showed 'transformer_encoder' as a submodule
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    # Register the hook
    # We want the output of the transformer encoder, which should be (batch, seq_len, dim)
    # The module name 'transformer_encoder' was seen in the inspection
    if hasattr(model, 'transformer_encoder'):
        print("Registering hook on transformer_encoder...")
        model.transformer_encoder.register_forward_hook(get_activation('transformer_encoder'))
    else:
        # Fallback inspection if structure is slightly different
        print("Warning: transformer_encoder not found directly. Printing named children:")
        for name, _ in model.named_children():
            print(name)
        return

    # Run a prediction to trigger the forward pass
    print("Running prediction...")
    # predict_proba usually triggers the full forward pass
    classifier.predict_proba(X)
    
    # Check if we captured something
    if 'transformer_encoder' in activation:
        latents = activation['transformer_encoder']
        print(f"Captured latents shape: {latents.shape}")
        
        # Verify dimension is 192
        assert latents.shape[-1] == 192, f"Expected last dimension 192, got {latents.shape[-1]}"
        print("SUCCESS: Latent extraction verified with dimension 192.")
        
        # Verify batch size matches 
        # Note: TabPFN might process in chunks or ensembles, so exact shape might need interpretation
        # but for n_estimators=1 and small data, it should relate to input size.
    else:
        print("FAILED: No activation captured.")

if __name__ == "__main__":
    test_latent_extraction()
