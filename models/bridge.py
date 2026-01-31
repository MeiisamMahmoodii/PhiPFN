
import torch
import torch.nn as nn

class LinearBridge(nn.Module):
    """
    A simple linear bridge to project TabPFN embeddings to LLM dimension.
    
    Args:
        input_dim (int): Dimension of TabPFN embeddings (default 192).
        output_dim (int): Dimension of Phi-4/LLM embeddings (default 3072).
    """
    def __init__(self, input_dim=192, output_dim=3072):
        super().__init__()
        self.bridge = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Ensembles, Batch, Dim) or (Batch, Dim)
            
        Returns:
            Tensor of shape (Batch, output_dim)
        """
        # If input is (Ensembles, Batch, Dim), average over ensembles
        if x.dim() == 3:
            # Check if likely (Ensemble, Batch, Dim) or (Batch, Seq, Dim)
            # TabPFN extraction gave (8, 100, 192) for 100 samples.
            # We assume ensemble dimension is 0.
            x = x.mean(dim=0)
            
        return self.bridge(x)

if __name__ == "__main__":
    # verification code
    model = LinearBridge()
    dummy_input = torch.randn(2, 10, 192)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 10, 3072)
    print("Shape verification passed")
