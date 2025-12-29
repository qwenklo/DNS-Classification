"""
Test ONNX model inference and compare with original PyTorch model.
"""

import torch
import yaml
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from dns_model import DNSModel
from dns_preprocess import load_preprocessed_data
import time


def test_onnx_model(config_path="config.yaml", model_path="DNS-Assignment2025.pth"):
    """Test ONNX model inference and compare with PyTorch model."""
    print("=" * 60)
    print("Testing ONNX Model")
    print("=" * 60)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    print("Loading test data...")
    train, test, val, class_names = load_preprocessed_data(config_path)
    X_test = test[:, :-1].astype(np.float32)
    y_test = test[:, -1].astype(np.int64)
    
    print(f"Test set size: {len(X_test)} samples")
    print(f"Number of features: {X_test.shape[1]}")
    
    # Load scaler
    scaler_path = Path("scaler.pkl")
    if not scaler_path.exists():
        raise FileNotFoundError("Scaler not found. Please run DNSAssignment.py first.")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    X_test_scaled = scaler.transform(X_test)
    
    # Load original PyTorch model
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    print(f"\nLoading PyTorch model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    input_features = X_test.shape[1]
    num_classes = len(class_names)
    
    nn_config = config.get('model', {}).get('neural_network', {})
    hidden_sizes = nn_config.get('hidden_layers', [256, 128])  # Default matches DNSAssignment.py
    dropout = nn_config.get('dropout', 0.2)  # Default matches DNSAssignment.py
    
    pytorch_model = DNSModel(
        input_features=input_features,
        num_classes=num_classes,
        hidden_sizes=hidden_sizes,
        dropout=dropout
    ).to(device)
    
    pytorch_model.load_state_dict(torch.load(model_path, map_location=device))
    pytorch_model.eval()
    
    # Get PyTorch predictions
    print("Getting predictions from PyTorch model...")
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    with torch.inference_mode():
        pytorch_outputs = pytorch_model(X_test_tensor)
        pytorch_proba = torch.softmax(pytorch_outputs, dim=1).cpu().numpy()
        pytorch_predictions = np.argmax(pytorch_proba, axis=1)
    
    # Load ONNX model
    onnx_dir = Path(config['onnx']['output_path'])
    onnx_model_name = config['onnx']['model_name']
    onnx_path = onnx_dir / onnx_model_name
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}. Run convert.py first.")
    
    print(f"\nLoading ONNX model from {onnx_path}...")
    
    try:
        import onnxruntime as ort
        
        # Create inference session
        session = ort.InferenceSession(str(onnx_path))
        
        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"Input name: {input_name}")
        print(f"Output name: {output_name}")
        
        # Run inference
        print("Running ONNX inference...")
        onnx_outputs = session.run([output_name], {input_name: X_test_scaled})
        onnx_logits = onnx_outputs[0]
        
        # Apply softmax to get probabilities
        onnx_proba = np.exp(onnx_logits) / np.sum(np.exp(onnx_logits), axis=1, keepdims=True)
        onnx_predictions = np.argmax(onnx_proba, axis=1)
        
        # Compare results
        print("\n" + "=" * 60)
        print("Comparison Results")
        print("=" * 60)
        
        from sklearn.metrics import accuracy_score
        
        pytorch_accuracy = accuracy_score(y_test, pytorch_predictions)
        onnx_accuracy = accuracy_score(y_test, onnx_predictions)
        
        print(f"\nPyTorch Model Accuracy: {pytorch_accuracy*100:.4f}%")
        print(f"ONNX Model Accuracy: {onnx_accuracy*100:.4f}%")
        print(f"Difference: {abs(pytorch_accuracy - onnx_accuracy)*100:.6f}%")
        
        # Prediction agreement
        agreement = np.mean(pytorch_predictions == onnx_predictions)
        print(f"\nPrediction Agreement: {agreement:.4f} ({agreement*100:.2f}%)")
        
        # Probability difference
        proba_diff = np.mean(np.abs(pytorch_proba - onnx_proba))
        print(f"Average Probability Difference: {proba_diff:.6f}")
        
        # Test on a few samples
        print("\n" + "=" * 60)
        print("Sample Predictions")
        print("=" * 60)
        
        n_samples = min(5, len(X_test))
        print(f"\nFirst {n_samples} test samples:")
        
        for i in range(n_samples):
            true_label = class_names[y_test[i]]
            pytorch_pred = class_names[pytorch_predictions[i]]
            onnx_pred = class_names[onnx_predictions[i]]
            
            pytorch_conf = pytorch_proba[i][pytorch_predictions[i]]
            onnx_conf = onnx_proba[i][onnx_predictions[i]]
            
            match = "✓" if pytorch_pred == onnx_pred else "✗"
            print(f"\nSample {i+1}:")
            print(f"  True Label: {true_label}")
            print(f"  PyTorch: {pytorch_pred} (confidence: {pytorch_conf:.4f})")
            print(f"  ONNX: {onnx_pred} (confidence: {onnx_conf:.4f})")
            print(f"  Match: {match}")
        
        # Performance test
        print("\n" + "=" * 60)
        print("Performance Test")
        print("=" * 60)
        
        # PyTorch inference time
        test_samples = X_test_scaled[:100]
        test_tensor = torch.FloatTensor(test_samples).to(device)
        
        # Warmup
        with torch.inference_mode():
            _ = pytorch_model(test_tensor)
        
        start = time.time()
        for _ in range(10):
            with torch.inference_mode():
                _ = pytorch_model(test_tensor)
        pytorch_time = (time.time() - start) / 10
        
        # ONNX inference time
        start = time.time()
        for _ in range(10):
            _ = session.run([output_name], {input_name: test_samples})
        onnx_time = (time.time() - start) / 10
        
        print(f"\nPyTorch Model (100 samples): {pytorch_time*1000:.2f} ms")
        print(f"ONNX Model (100 samples): {onnx_time*1000:.2f} ms")
        if onnx_time > 0:
            print(f"Speedup: {pytorch_time/onnx_time:.2f}x")
        
        print("\n" + "=" * 60)
        print("ONNX Model Test Complete!")
        print("=" * 60)
        
    except ImportError:
        print("\n✗ Error: onnxruntime not installed.")
        print("Install with: pip install onnxruntime")
        raise
    except Exception as e:
        print(f"\n✗ Error testing ONNX model: {e}")
        raise


def test_single_prediction(config_path="config.yaml", sample_data=None):
    """Test ONNX model with a single sample."""
    print("\n" + "=" * 60)
    print("Single Sample Prediction Test")
    print("=" * 60)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load metadata
    onnx_dir = Path(config['onnx']['output_path'])
    metadata_path = onnx_dir / "model_metadata.pkl"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found. Run convert.py first.")
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        input_features = metadata['input_features']
        class_names = metadata['class_names']
    
    # Load scaler
    scaler_path = Path("scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Use provided sample or create dummy sample
    if sample_data is None:
        print("Creating dummy sample...")
        sample_data = np.random.randn(input_features).astype(np.float32)
    else:
        if len(sample_data) != input_features:
            raise ValueError(f"Sample data must have {input_features} features")
        sample_data = np.array(sample_data).astype(np.float32)
    
    # Scale sample
    sample_scaled = scaler.transform(sample_data.reshape(1, -1))
    
    # Load ONNX model
    onnx_model_name = config['onnx']['model_name']
    onnx_path = onnx_dir / onnx_model_name
    
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Run inference
        output = session.run([output_name], {input_name: sample_scaled})
        logits = output[0][0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        
        # Get prediction
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        print(f"\nPrediction: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"\nAll class probabilities:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {probabilities[i]:.4f}")
        
        return predicted_class, probabilities
        
    except ImportError:
        print("\n✗ Error: onnxruntime not installed.")
        print("Install with: pip install onnxruntime")
        raise


if __name__ == "__main__":
    try:
        test_onnx_model()
        test_single_prediction()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
