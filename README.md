# DNS Classification Project - Deep Learning Assignment

A **PyTorch-based Deep Learning** project for classifying DNS traffic into **benign**, **malware**, **phishing**, and **spam** categories.

## Overview

This project implements a complete Deep Learning pipeline for DNS traffic classification using PyTorch. It features a Multi-Layer Perceptron (MLP) neural network with batch normalization, dropout regularization, and comprehensive training infrastructure including WandB integration and ONNX export.

## Features

- **PyTorch Neural Network**: MLP with BatchNorm, GELU activation, and Dropout
- **Multi-class Classification**: Classifies DNS traffic into 4 categories (benign, malware, phishing, spam)
- **Complete Training Pipeline**: Early stopping, learning rate scheduling, gradient clipping
- **WandB Integration**: Automatic experiment tracking and visualization
- **ONNX Export**: Convert trained models to ONNX format for production deployment
- **Comprehensive Evaluation**: Detailed metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)

## Project Structure

```
DNS2021/
├── Data/
│   └── DNS2021/
│       ├── csv/                                    # Raw CSV files (downloaded)
│       │   ├── CSV_benign.csv
│       │   ├── CSV_malware.csv
│       │   ├── CSV_phishing.csv
│       │   └── CSV_spam.csv
│       ├── preprocess_csv.py                       # Data preprocessing script ⭐
│       ├── train.npy                               # Training data (features + labels)
│       ├── test.npy                                 # Test data
│       ├── val.npy                                  # Validation data
│       └── class_names.npy                          # Class names array
│
├── DNSAssignment.py                                 # Main training script ⭐
├── model.py                                         # Neural network architecture + sklearn models
├── dns_model.py                                     # PyTorch DNSModel (alternative)
├── dns_preprocess.py                                # PyTorch DataLoader creation ⭐
├── convert.py                                       # PyTorch to ONNX conversion ⭐
├── onnxtest.py                                      # ONNX model testing ⭐
├── preprocess.py                                    # Alternative preprocessing (sklearn-based)
├── train_mlp.py                                     # Alternative training script
├── config.yaml                                      # Training configuration ⭐
├── DNS-Assignment2025.pth                           # Trained model weights (generated)
├── DNS-Assignment2025.onnx                          # ONNX model file (generated)
├── scaler.pkl                                       # Saved StandardScaler (generated)
├── requirements.txt                                 # Python dependencies ⭐
├── README.md                                        # Project overview
├── README_SETUP.md                                  # Setup instructions
├── ARCHITECTURE_ANALYSIS.md                         # Architecture justification
├── Documentation.md                                 # This file
├── wandb/                                           # WandB experiment logs (generated)
└── models/                                          # Additional model checkpoints (generated)
```

## Quick Start

### 1. Install Dependencies

**Install PyTorch:**
```bash
# CPU only
pip install torch torchvision

# GPU support (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Install other dependencies:**
```bash
pip install pandas numpy scikit-learn pyyaml joblib tqdm wandb onnx onnxruntime torchsummary
```

### 2. Configure the Project

Edit `config.yaml` to set your data paths and training parameters. Optionally add your WandB API key for experiment tracking.

### 3. Run Training Pipeline

**Option A: All-in-one (Recommended)**
```bash
python DNSAssignment.py
```
This automatically:
- Checks for preprocessed data (runs preprocessing if needed)
- Creates PyTorch DataLoaders
- Trains the neural network
- Evaluates on test set
- Saves the model

**Option B: Step-by-step**
```bash
# 1. Preprocess data (if not already done)
cd Data/DNS2021
python preprocess_csv.py
cd ../..

# 2. Train model
python DNSAssignment.py

# 3. Convert to ONNX (optional)
python convert.py

# 4. Test ONNX model (optional)
python onnxtest.py
```

### Output Files

After training, you'll have:
- `DNS-Assignment2025.pth`: Trained PyTorch model
- `scaler.pkl`: Feature scaler for inference
- WandB logs: Training metrics and visualizations

## Model Architecture

The DNS Classification model uses a **Multi-Layer Perceptron (MLP)** architecture, which is optimal for tabular/feature-based data from CSV files.

**Why MLP?**
- ✅ Best suited for **tabular data** with engineered features (32 features from CIC-Bell-DNS 2021)
- ✅ Efficient training and inference for large datasets (400K+ samples)
- ✅ Can learn complex non-linear relationships between features
- ✅ Simpler and faster than RNN/CNN for feature-based classification

**For PCAP file analysis**, consider CNN or LSTM architectures (see `ARCHITECTURE_ANALYSIS.md` for details).

The model architecture:

```
Input (n_features)
  ↓
Linear(256) → BatchNorm → GELU → Dropout(0.2)
  ↓
Linear(128) → BatchNorm → GELU → Dropout(0.2)
  ↓
Output (4 classes: benign, malware, phishing, spam)
```

**Default Configuration:**
- Hidden layers: `[256, 128]` (configurable in `config.yaml`)
- Dropout: `0.2` (configurable in `config.yaml`)
- Batch size: `128`
- Learning rate: `0.001`
- Optimizer: `AdamW`
- Loss function: `CrossEntropyLoss`

**Key Features:**
- Batch Normalization for stable training
- GELU activation function
- Dropout regularization (20%)
- Gradient clipping (max_norm=1.0)
- Early stopping (patience=5)
- Learning rate scheduling (ReduceLROnPlateau)

## Configuration

The `config.yaml` file contains all configuration settings for the **CIC-Bell-DNS 2021 dataset**:

- **Data paths**: Locations of CSV files (benign, malware, phishing, spam)
- **Model architecture**: Hidden layer sizes, dropout
- **Training**: Batch size, learning rate, epochs
- **WandB**: Project name and API key (optional)
- **ONNX export**: Export configuration
- **Class labels**: 4 classes (benign, malware, phishing, spam)

**Recommended Settings for CIC-Bell-DNS 2021:**
- `batch_size: 128` - Good for large dataset (400K+ samples)
- `learning_rate: 0.001` - Conservative rate for stable training
- `num_epochs: 50` - Maximum epochs (early stopping may stop earlier around 20-30 epochs)
- `hidden_layers: [256, 128]` - Good balance for feature count
- `dropout: 0.2` - Standard regularization

**Dataset Reference:** [CIC-Bell-DNS 2021](https://www.unb.ca/cic/datasets/dns-2021.html)

See `Documentation.md` for detailed configuration options.

## Training Features

- **Early Stopping**: Stops training if validation loss doesn't improve (patience=5)
- **Learning Rate Scheduling**: Automatically reduces learning rate (ReduceLROnPlateau)
- **WandB Integration**: Tracks training metrics, loss curves, confusion matrix
- **GPU Support**: Automatic CUDA detection and usage
- **Model Checkpointing**: Saves best model based on validation loss
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
- **First Trial Setup**: Basic configuration for initial training (50 epochs, no weighted sampler)

## Evaluation Metrics

The training script provides:
- **Accuracy**: Overall classification accuracy
- **F1 Scores**: Macro and Weighted F1 scores
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Per-class prediction breakdown
- **WandB Logging**: Automatic logging of:
  - Train Loss (per epoch)
  - Validation Loss (per epoch)
  - Learning Rate
  - Test Accuracy
  - Test F1 Scores
  - Confusion Matrix

## Requirements

- Python 3.7+ (Python 3.8+ recommended)
- PyTorch 1.9+
- pandas, numpy, scikit-learn
- pyyaml, joblib, tqdm
- wandb (for experiment tracking)
- onnx, onnxruntime (for ONNX conversion and testing)
- torchsummary (for model architecture visualization)

**Optional:**
- CUDA-capable GPU for faster training


## Contact

**Dataset & Research Institution**

For questions related to the dataset, data collection methodology, or licensing:

**Canadian Institute for Cybersecurity (CIC)**
University of New Brunswick (UNB)
Website: https://www.unb.ca/cic

Dataset page: https://www.unb.ca/cic/datasets/dns-2021.html

## Wandb Evaluation

<img width="3160" height="1660" alt="W B Chart 29_12_2025, 3_55_54 pm" src="https://github.com/user-attachments/assets/1619289e-6a2e-4edd-b0dd-465616e57312" />

<img width="3160" height="1660" alt="W B Chart 29_12_2025, 3_56_58 pm" src="https://github.com/user-attachments/assets/b6f230b5-4a97-45b1-8eb4-f6ba4da9f8ba" />

<img width="3160" height="1660" alt="W B Chart 29_12_2025, 3_56_58 pm" src="https://github.com/user-attachments/assets/1dc4fddb-e40b-48b5-9761-a7c60965120b" />

<img width="3160" height="1660" alt="W B Chart 29_12_2025, 3_57_48 pm" src="https://github.com/user-attachments/assets/9ce7d62c-5754-4bd3-b6d1-b0230d3f1520" />









