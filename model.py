"""
Machine learning model definitions for DNS classification.
Contains both PyTorch (DNSModel) and sklearn-based models (DNSClassifier).
"""

import pandas as pd
import numpy as np
import yaml
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import joblib

# Import PyTorch DNSModel for compatibility
try:
    import torch
    import torch.nn as nn
    
    class DNSModel(nn.Module):
        """
        Feedforward Neural Network (MLP) for DNS Classification.
        
        Architecture:
        - Input Layer: n_features (tabular data from CSV)
        - Hidden Layer 1: Linear(256) -> BatchNorm -> GELU -> Dropout(0.2)
        - Hidden Layer 2: Linear(128) -> BatchNorm -> GELU -> Dropout(0.2)
        - Output Layer: Linear(num_classes) -> 4 classes (benign, malware, phishing, spam)
        
        Justification:
        - MLP is optimal for tabular/feature-based data (not sequential/image)
        - BatchNorm: Stabilizes training, normalizes activations
        - GELU: Smooth activation, better gradients than ReLU
        - Dropout: Prevents overfitting on large dataset
        - See ARCHITECTURE_ANALYSIS.md for detailed justification
        """
        
        def __init__(self, input_features, num_classes=4, hidden_sizes=[256, 128], dropout=0.2):
            """
            Initialize the DNS Classification Model.
            
            Args:
                input_features (int): Number of input features
                num_classes (int): Number of output classes (default: 4 for benign, malware, phishing, spam)
                hidden_sizes (list): List of hidden layer sizes (default: [256, 128])
                dropout (float): Dropout probability (default: 0.2)
            """
            super(DNSModel, self).__init__()
            
            self.input_features = input_features
            self.num_classes = num_classes
            
            # First hidden layer
            self.fc1 = nn.Linear(input_features, 256)
            self.batch_norm1 = nn.BatchNorm1d(256)
            self.activation1 = nn.GELU()
            self.dropout1 = nn.Dropout(dropout)  # Use parameter instead of hardcoded
            
            # Second hidden layer
            self.fc2 = nn.Linear(256, 128)
            self.batch_norm2 = nn.BatchNorm1d(128)
            self.activation2 = nn.GELU()
            self.dropout2 = nn.Dropout(dropout)  # Use parameter instead of hardcoded
            
            # Output layer
            self.fc3 = nn.Linear(128, num_classes)
        
        def forward(self, x):
            """
            Forward pass through the network.
            
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_features)
                
            Returns:
                torch.Tensor: Output logits of shape (batch_size, num_classes)
            """
            x = self.fc1(x)
            x = self.batch_norm1(x)
            x = self.activation1(x)
            x = self.dropout1(x)
            
            x = self.fc2(x)
            x = self.batch_norm2(x)
            x = self.activation2(x)
            x = self.dropout2(x)
            
            x = self.fc3(x)
            return x
        
        def get_model_info(self):
            """Get model information."""
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            return {
                'input_features': self.input_features,
                'num_classes': self.num_classes,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
            }
    
except ImportError:
    # PyTorch not available, DNSModel will not be available
    DNSModel = None


class DNSClassifier:
    """
    DNS classification model using sklearn-based algorithms.
    Supports Random Forest, XGBoost, and sklearn MLPClassifier.
    
    Note: For PyTorch-based training, use DNSModel from dns_model.py or this module.
    """
    
    def __init__(self, config_path="config.yaml"):
        """Initialize classifier with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        # Support both old nested and new flat config structure
        if 'model' in self.config and 'type' in self.config['model']:
            self.model_type = self.config['model']['type']
            self.model_name = self.config['model'].get('name', 'dns_classifier')
        else:
            # Default to neural_network if not specified
            self.model_type = "neural_network"
            self.model_name = "dns_classifier"
        
    def build_model(self):
        """Build the model based on configuration."""
        print(f"Building {self.model_type} model...")
        
        if self.model_type == "random_forest":
            params = self.config.get('model', {}).get('random_forest', {})
            self.model = RandomForestClassifier(**params)
            
        elif self.model_type == "xgboost":
            params = self.config.get('model', {}).get('xgboost', {})
            self.model = XGBClassifier(**params)
            
        elif self.model_type == "neural_network":
            from sklearn.neural_network import MLPClassifier
            params = self.config.get('model', {}).get('neural_network', {})
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(params.get('hidden_layers', [256, 128])),
                activation=params.get('activation', 'relu'),
                learning_rate_init=params.get('learning_rate', 0.001),
                max_iter=params.get('epochs', 50),
                batch_size=params.get('batch_size', 32),
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"Model built: {self.model}")
        return self.model
    
    def load_data(self):
        """Load preprocessed data."""
        print("Loading preprocessed data...")
        # Support both old nested and new flat config structure
        if 'data' in self.config and 'output_dir' in self.config['data']:
            output_dir = Path(self.config['data']['output_dir'])
        else:
            output_dir = Path('processed_data')
        
        X_train = pd.read_csv(output_dir / "X_train.csv")
        X_val = pd.read_csv(output_dir / "X_val.csv")
        X_test = pd.read_csv(output_dir / "X_test.csv")
        
        y_train = np.load(output_dir / "y_train.npy")
        y_val = np.load(output_dir / "y_val.npy")
        y_test = np.load(output_dir / "y_test.npy")
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        print("Training model...")
        
        if X_val is not None and y_val is not None:
            # Use validation set for early stopping if supported
            if self.model_type == "xgboost":
                verbose = self.config.get('training', {}).get('verbose', True)
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=verbose
                )
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        print("Training complete!")
        return self.model
    
    def evaluate(self, X, y, set_name="Test"):
        """Evaluate the model."""
        print(f"\nEvaluating on {set_name} set...")
        
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        print(f"{set_name} Accuracy: {accuracy:.4f}")
        
        # Get class names
        if 'data' in self.config and 'output_dir' in self.config['data']:
            output_dir = Path(self.config['data']['output_dir'])
        else:
            output_dir = Path('processed_data')
            
        with open(output_dir / "preprocessor.pkl", 'rb') as f:
            preprocessor_data = pickle.load(f)
            label_encoder = preprocessor_data['label_encoder']
            class_names = label_encoder.classes_
        
        print(f"\n{classification_report(y, y_pred, target_names=class_names)}")
        print(f"\nConfusion Matrix:\n{confusion_matrix(y, y_pred)}")
        
        return {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_true': y
        }
    
    def save_model(self):
        """Save the trained model."""
        save_model = self.config.get('training', {}).get('save_model', True)
        if not save_model:
            return
        
        if 'model' in self.config and 'save_path' in self.config['model']:
            model_dir = Path(self.config['model']['save_path'])
        else:
            model_dir = Path('models')
        
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"{self.model_name}_{self.model_type}.pkl"
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path=None):
        """Load a saved model."""
        if model_path is None:
            if 'model' in self.config and 'save_path' in self.config['model']:
                model_dir = Path(self.config['model']['save_path'])
            else:
                model_dir = Path('models')
            model_path = model_dir / f"{self.model_name}_{self.model_type}.pkl"
        
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return self.model
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict_proba(X)


def main():
    """Main training pipeline for sklearn models."""
    print("=" * 50)
    print("DNS Classification Model Training (sklearn)")
    print("=" * 50)
    
    # Initialize classifier
    classifier = DNSClassifier()
    
    # Build model
    classifier.build_model()
    
    # Load data
    data = classifier.load_data()
    
    # Train model
    classifier.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    # Evaluate on validation set
    classifier.evaluate(data['X_val'], data['y_val'], "Validation")
    
    # Evaluate on test set
    classifier.evaluate(data['X_test'], data['y_test'], "Test")
    
    # Save model
    classifier.save_model()
    
    print("\nTraining pipeline complete!")


if __name__ == "__main__":
    main()
