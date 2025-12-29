"""
MLP Training Script for CIC-Bell-DNS-2021 Dataset
Uses preprocessed data from Data/DNS2021/preprocess_csv.py
First Training Trial - 50 Epochs with WandB Tracking
"""

import torch
import numpy as np
from dns_preprocess import preprocess
import torch.optim as optim
from model import DNSModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torchsummary import summary
import wandb
import yaml
import os

# =========================
# CONFIG
# =========================
WANDB_KEY = "f16133b01f4cb7007f0ee94a4c59eeff475240f9" 
WANDB_PROJECT = "DNS-Assignment2025"

# Data paths (relative to script location)
DATA_DIR = "Data/DNS2021"
TRAIN_PATH = os.path.join(DATA_DIR, "train.npy")
VAL_PATH = os.path.join(DATA_DIR, "val.npy")
TEST_PATH = os.path.join(DATA_DIR, "test.npy")
CLASS_NAMES_PATH = os.path.join(DATA_DIR, "class_names.npy")
CONFIG_PATH = "config.yaml"
MODEL_SAVE_PATH = "DNS-Assignment2025.pth"
SCALER_SAVE_PATH = "scaler.pkl"

# =========================
# FUNCTIONS
# =========================
def train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs):
    """
    Trains the model for a given number of epochs.
    
    Args:
        model: The model to train.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        device: The device to train on.
        criterion: The loss function.
        optimizer: The optimizer.
        scheduler: The scheduler.
        num_epochs: The number of epochs to train for.

    Returns:
        The trained model.
    """
    best_loss = float('inf')
    patience = 5
    no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data, target = data.to(device), target.to(device).long()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, _, _ = evaluate_model(model, val_loader, device, criterion)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✓ Model saved to {MODEL_SAVE_PATH} (val_loss: {val_loss:.6f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}/{num_epochs}")
                print(f"Best validation loss: {best_loss:.6f}")
                break

        scheduler.step(val_loss)
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "Train Loss": train_loss, 
                "Val Loss": val_loss,
                "Learning Rate": optimizer.param_groups[0]['lr'],
                "Epoch": epoch + 1
            })
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    return model

def evaluate_model(model, dataloader, device, criterion=None):
    """
    Evaluates the model on a given dataloader.
    Can compute loss and/or return predictions and labels.

    Args:
        model: The model to evaluate.
        dataloader: The data loader to evaluate on.
        device: The device to evaluate on.
        criterion: The loss function.

    Returns:
        The average loss, all predictions, and all labels.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for samples, labels in dataloader:
            samples = samples.to(device).float()
            labels = labels.to(device).long()

            logits = model(samples)

            if criterion is not None:
                loss = criterion(logits, labels)
                bs = labels.size(0)
                total_loss += loss.item() * bs      # weight by batch size
                total_samples += bs

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = (total_loss / total_samples) if (criterion is not None and total_samples > 0) else 0.0
    return avg_loss, all_preds, all_labels

def test_and_report(model, test_loader, device, class_names):
    """
    Evaluates a model on the test set and prints a classification report
    and confusion matrix.

    Args:
        model: The model to evaluate.
        test_loader: The test data loader.
        device: The device to evaluate on.
        class_names: The class names.

    Returns:
        The accuracy of the model on the test set.
    """
    print("\n" + "="*60)
    print("--- Final Test Evaluation ---")
    print("="*60)
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.inference_mode():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)
            labels = labels.to(device).long()
            
            predictions = model(samples)
            preds = torch.argmax(predictions, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\n✓ Final Test Accuracy: {acc*100:.2f}%")
    print(f"✓ Test F1 Score (Macro): {f1_macro*100:.2f}%")
    print(f"✓ Test F1 Score (Weighted): {f1_weighted*100:.2f}%")

    # Log to wandb
    if wandb.run is not None:
        wandb.log({
            "Test Accuracy": acc*100,
            "Test F1 Macro": f1_macro*100,
            "Test F1 Weighted": f1_weighted*100
        })

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Log confusion matrix to wandb
    if wandb.run is not None:
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=class_names.tolist()
        )})
    
    return acc

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("="*60)
    print("CIC-Bell-DNS-2021 MLP Training - First Trial")
    print("="*60)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nConfig loaded:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Epochs: {config['num_epochs']}")
    
    # Load preprocessed data
    print(f"\nLoading preprocessed data...")
    print(f"  Train: {TRAIN_PATH}")
    print(f"  Val: {VAL_PATH}")
    print(f"  Test: {TEST_PATH}")
    print(f"  Class names: {CLASS_NAMES_PATH}")
    
    train = np.load(TRAIN_PATH, allow_pickle=True)
    val = np.load(VAL_PATH, allow_pickle=True)
    test = np.load(TEST_PATH, allow_pickle=True)
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
    
    num_classes = len(class_names)
    num_features = train.shape[1] - 1
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  Training samples: {len(train):,}")
    print(f"  Validation samples: {len(val):,}")
    print(f"  Test samples: {len(test):,}")
    print(f"  Number of features: {num_features}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {class_names.tolist()}")
    
    # Verify label range
    train_labels = train[:, -1].astype(np.int64)
    val_labels = val[:, -1].astype(np.int64)
    test_labels = test[:, -1].astype(np.int64)
    
    max_label = max(train_labels.max(), val_labels.max(), test_labels.max())
    min_label = min(train_labels.min(), val_labels.min(), test_labels.min())
    
    print(f"\n  Label range: {min_label} to {max_label} (expected: 0 to {num_classes-1})")
    
    if max_label >= num_classes or min_label < 0:
        raise ValueError(f"Labels out of range! Labels are in [{min_label}, {max_label}], but model expects [0, {num_classes-1}]")
    
    # Preprocess data (create DataLoaders)
    print(f"\n--- Creating DataLoaders ---")
    train_loader, test_loader, val_loader = preprocess(
        train, test, val, 
        batch_size=config['batch_size'], 
        scaler_save_path=SCALER_SAVE_PATH,
        use_weighted_sampler=False  # First trial: no weighted sampler
    )
    
    # Create model
    print(f"\n--- Creating MLP Model ---")
    model = DNSModel(
        input_features=num_features, 
        num_classes=num_classes,
        hidden_sizes=[256, 128],
        dropout=0.2
    ).to(device)
    
    # Print model summary
    print("\nModel Architecture:")
    print(summary(model, input_size=(num_features,), device=device))
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Initialize wandb
    try:
        wandb.login(key=WANDB_KEY)
        wandb.init(
            project=WANDB_PROJECT,
            name=f"First_Trial_epochs{config['num_epochs']}_lr{config['learning_rate']}",
            config={
                "batch_size": config['batch_size'],
                "num_epochs": config['num_epochs'],
                "learning_rate": config['learning_rate'],
                "optimizer": "AdamW",
                "scheduler": "ReduceLROnPlateau",
                "model": "MLP",
                "hidden_sizes": [256, 128],
                "dropout": 0.2,
                "num_features": num_features,
                "num_classes": num_classes,
                "classes": class_names.tolist()
            },
            tags=["First-Trial", "MLP", "DNS-Classification"]
        )
        wandb.watch(model, log="all")
        print("\n✓ WandB initialized")
        print(f"  Project: {WANDB_PROJECT}")
        print(f"  Run URL: {wandb.run.url if wandb.run else 'N/A'}")
    except Exception as e:
        print(f"\n⚠ WandB initialization failed: {e}")
        print("  Continuing without WandB logging...")
    
    # Train model
    print(f"\n--- Starting Training ---")
    model = train_model(
        model, train_loader, val_loader, 
        device, criterion, optimizer, scheduler, 
        num_epochs=config['num_epochs']
    )
    
    # Test model
    test_acc = test_and_report(model, test_loader, device, class_names)
    
    # Final summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Scaler saved to: {SCALER_SAVE_PATH}")
    print(f"Final test accuracy: {test_acc*100:.2f}%")
    print("="*60)
    
    if wandb.run is not None:
        wandb.finish()
