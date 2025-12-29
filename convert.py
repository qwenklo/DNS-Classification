import torch
from model import DNSModel
import numpy as np
import yaml
from pathlib import Path
import pickle
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load preprocessed data to get feature count
output_dir = Path('processed_data')
X_train_path = output_dir / "X_train.csv"

if not X_train_path.exists():
    raise FileNotFoundError(f"Preprocessed data not found at {X_train_path}. Please run preprocess.py first.")

# Load training data to get number of features
X_train = pd.read_csv(X_train_path)
num_features = X_train.shape[1]

# Load label encoder to get class names
preprocessor_path = output_dir / "preprocessor.pkl"
if not preprocessor_path.exists():
    raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

with open(preprocessor_path, 'rb') as f:
    preprocessor_data = pickle.load(f)
    label_encoder = preprocessor_data['label_encoder']
    class_names = label_encoder.classes_

num_classes = len(class_names)

print(f"Finished loading data. Features: {num_features}, Classes: {num_classes}")

model_path = 'DNS-Assignment2025.pth'
model = DNSModel(input_features=num_features, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

print("\nExporting final model to ONNX")
onnx_path = model_path.replace(".pth", ".onnx")

dummy_input = torch.randn(size=(1, num_features), device=device, dtype=torch.float32)

model.eval()
try:
    torch.onnx.export(
        model.to(device),
        dummy_input, onnx_path, verbose=False,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Final model exported to ONNX at: {onnx_path}")
except Exception as e:
    print(f"Error exporting model to ONNX: {e}")
