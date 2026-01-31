
import numpy as np
from tabpfn import TabPFNClassifier
import torch

print("Initializing TabPFNClassifier...")
# Updated for TabPFN v2 API
classifier = TabPFNClassifier(device='cpu', n_estimators=2)

X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
y = np.array([1, 1, 0, 0])

print("Fitting...")
classifier.fit(X, y)
print("Predicting...")
preds = classifier.predict(X)
print(f"Predictions: {preds}")

print("\n--- Model Architecture Inspection ---")
if hasattr(classifier, 'model_'):
    print("Found 'model_' property.")
    print(classifier.model_)
    # Also inspect what's inside the model
    # Usually it has an encoder, or transformer backbone
    for name, module in classifier.model_.named_children():
        print(f"Submodule: {name} -> {type(module)}")

elif hasattr(classifier, 'models_'):
    print("Found 'models_' attribute.")
    print(classifier.models_[0])
else:
    print("Could not find direct model attribute. Listing attributes:")
    print(dir(classifier))
