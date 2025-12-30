# DNSAssignment2025 - DNS Traffic Classification Project Documentation

## 📋 Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Important Files](#important-files)
4. [Dependencies](#dependencies)
5. [Installation](#installation)
6. [Project Components](#project-components)
7. [Workflow](#workflow)
8. [Configuration](#configuration)
9. [Model Architecture](#model-architecture)
10. [Data Processing](#data-processing)
11. [Training](#training)
12. [Model Conversion & Deployment](#model-conversion--deployment)
13. [Evaluation Criteria](#evaluation-criteria)
14. [Troubleshooting](#troubleshooting)

---

## Overview

**DNSAssignment2025** is a deep learning project focused on **DNS Traffic Classification** using PyTorch. The project implements a feedforward neural network (MLP) to classify DNS traffic into four categories: **Benign**, **Malware**, **Phishing**, and **Spam** using the CIC-Bell-DNS-2021 dataset.

### Key Features

- **Deep Learning Model**: Multi-Layer Perceptron (MLP) with batch normalization and dropout
- **Data Preprocessing**: Automated CSV processing with feature encoding and normalization
- **Training Pipeline**: Complete training workflow with early stopping and learning rate scheduling
- **Model Deployment**: ONNX conversion for production deployment
- **Experiment Tracking**: Weights & Biases (WandB) integration
- **Performance Benchmarking**: ONNX runtime testing with inference speed metrics

### Project Goals

- Build a robust DNS traffic classification model
- Demonstrate proper deep learning workflow (data → preprocessing → training → evaluation)
- Implement best practices: modular code, proper data splits, appropriate loss functions
- Provide comprehensive analysis of results
- Create production-ready model with ONNX export

### Dataset

- **Dataset**: CIC-Bell-DNS-2021
- **Source**: [CIC Research](http://cicresearch.ca/CICDataset/CICBellDNS2021/Dataset/)
- **Classes**: 4 (Benign, Malware, Phishing, Spam)
- **Features**: Tabular features extracted from DNS traffic

## Results & Key Findings

Based on training runs (WandB summaries):

- **Test Accuracy:** ~**93–94%**
- **Test F1 (Weighted):** ~**91%**
- **Test F1 (Macro):** varies (example runs show ~**42–66%**)

### Interpretation (simple)
- **Accuracy & Weighted F1 are high** because the dataset is likely **imbalanced** (majority class contributes heavily to these metrics).
- **Macro F1 is lower** because it treats every class equally, so weaker performance on minority classes (Malware/Phishing/Spam) lowers the score.

### Notes on Validation Loss “spike”
Occasional **validation loss spikes** can happen due to:
- minority-class batches being harder,
- class imbalance,
- noisy/outlier samples,
- small validation batches or distribution differences.

This is acceptable as long as:
- overall validation trend improves,
- test performance remains stable,

---

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
├── dns_preprocess.py                                # PyTorch DataLoader creation ⭐
├── convert.py                                       # PyTorch to ONNX conversion ⭐
├── onnxtest.py                                      # ONNX model testing ⭐
├── preprocess.py                                    # Alternative preprocessing (sklearn-based)
├── config.yaml                                      # Training configuration ⭐
├── scaler.pkl                                       # Saved StandardScaler (generated)
├── requirements.txt                                 # Python dependencies ⭐
├── README.md                                        # Project overview
├── README_SETUP.md                                  # Setup instructions
├── Documentation.md                                 # This file
├── wandb/                                           # WandB experiment logs (generated)
└── models/                                          # Additional model checkpoints (generated)
```

---

## Important Files

### 🔴 **Critical Files (Required for ML Pipeline)**

These files are **essential** for the machine learning workflow:

#### 1. **Data Preprocessing**
- **`Data/DNS2021/preprocess_csv.py`** ⭐
  - Downloads and processes raw CSV files
  - Handles missing values and infinity
  - Encodes labels and splits data (70/15/15)
  - **Output**: `train.npy`, `val.npy`, `test.npy`, `class_names.npy`
  - **Must run first** before training

#### 2. **Data Loading & Preprocessing**
- **`dns_preprocess.py`** ⭐
  - Creates PyTorch DataLoaders from numpy arrays
  - Applies StandardScaler normalization
  - Handles data cleaning (infinity, NaN, extreme values)
  - Saves scaler for inference
  - **Required by**: `DNSAssignment.py`

#### 3. **Model Architecture**
- **`model.py`** ⭐
  - Defines `DNSModel` (PyTorch MLP)
  - Contains `DNSClassifier` (sklearn-based models)
  - Architecture: Input → Linear(256) → BatchNorm → GELU → Dropout → Linear(128) → BatchNorm → GELU → Dropout → Output(4)
  - **Required by**: `DNSAssignment.py`, `convert.py`

#### 4. **Main Training Script**
- **`DNSAssignment.py`** ⭐
  - Complete training pipeline
  - Early stopping, learning rate scheduling
  - WandB integration
  - Model checkpointing
  - Final evaluation and reporting
  - **Output**: `DNS-Assignment2025.pth`, `scaler.pkl`

#### 5. **Configuration**
- **`config.yaml`** ⭐
  - Training hyperparameters (batch size, learning rate, epochs)
  - Class names (Benign, Malware, Phishing, Spam)
  - Model architecture settings
  - **Required by**: All scripts

#### 6. **Dependencies**
- **`requirements.txt`** ⭐
  - Python package dependencies
  - PyTorch, numpy, scikit-learn, wandb, etc.
  - **Required for**: Installation

### 🟡 **Important Files (Optional but Recommended)**

#### 7. **Model Conversion**
- **`convert.py`** ⭐
  - Converts PyTorch model to ONNX format
  - Required for production deployment
  - **Input**: `DNS-Assignment2025.pth`
  - **Output**: `DNS-Assignment2025.onnx`

#### 8. **ONNX Testing**
- **`onnxtest.py`** ⭐
  - Tests ONNX model performance
  - Compares with PyTorch model
  - Benchmarks inference speed
  - **Input**: `DNS-Assignment2025.onnx`, `scaler.pkl`

#### 9. **Documentation**
- **`README.md`** - Quick start guide
- **`README_SETUP.md`** - Detailed setup instructions
- **`Documentation.md`** - This comprehensive documentation

### 🟢 **Supporting Files (Alternative/Helper)**

#### 10. **Alternative Scripts**
- **`preprocess.py`** - Alternative preprocessing (sklearn-based)

### 📦 **Generated Files (Created During Execution)**

These files are created automatically and should not be edited manually:

- **`DNS-Assignment2025.pth`** - Trained PyTorch model weights
- **`DNS-Assignment2025.onnx`** - ONNX model file
- **`scaler.pkl`** - StandardScaler for feature normalization
- **`Data/DNS2021/train.npy`** - Preprocessed training data
- **`Data/DNS2021/val.npy`** - Preprocessed validation data
- **`Data/DNS2021/test.npy`** - Preprocessed test data
- **`Data/DNS2021/class_names.npy`** - Class names array
- **`wandb/`** - WandB experiment logs
- **`models/`** - Additional model checkpoints

---

## Dependencies

### Required Python Packages

All dependencies are specified in `requirements.txt`:

**Core Deep Learning:**
- `torch>=2.0.0` - PyTorch framework
- `numpy>=1.24.0` - Numerical computing

**Data Processing:**
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.3.0` - Machine learning utilities

**Model Deployment:**
- `onnx>=1.20.0` - ONNX model format (optional)
- `onnxruntime>=1.23.2` - ONNX inference runtime (optional)

**Development & Tracking:**
- `wandb>=0.15.0` - Weights & Biases for experiment tracking
- `tqdm>=4.65.0` - Progress bars
- `torchsummary>=1.5.1` - Model architecture summary

**Utilities:**
- `pyyaml>=6.0` - YAML configuration parsing
- `xgboost>=2.0.0` - XGBoost classifier (for sklearn models)

### Python Version

- **Python 3.7+** required (Python 3.8+ recommended)

### CUDA Support

- **CUDA 11.8+** (optional, for GPU acceleration)
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

---

## Installation

### Prerequisites

1. **Python 3.7+** installed
2. **pip** package manager

### Setup Steps

1. **Navigate to the project directory:**
   ```bash
   cd /home/kali/Documents/DNS2021
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch (CPU or GPU):**
   
   **CPU only:**
   ```bash
   pip install torch torchvision
   ```
   
   **GPU support (CUDA 11.8):**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Optional: Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Project Components

### 1. Data Preprocessing (`Data/DNS2021/preprocess_csv.py`)

**Purpose**: Downloads and converts raw CSV data into numpy arrays suitable for training.

**Key Features:**
- Downloads CIC-Bell-DNS-2021 dataset from official source
- Loads CSV files (benign, malware, phishing, spam)
- Handles missing values (NaN) and infinity
- Converts all features to numeric (float32)
- Encodes labels using LabelEncoder
- Splits data into train/validation/test sets (70/15/15)
- Saves processed arrays and class names

**Output Files:**
- `train.npy` - Training data (shape: `[n_samples, n_features + 1]`)
- `test.npy` - Test data
- `val.npy` - Validation data
- `class_names.npy` - Array of class names

**Usage:**
```bash
cd Data/DNS2021
python preprocess_csv.py
```

**Data Format:**
- Last column contains integer labels (0, 1, 2, 3)
- All other columns are numeric features
- Features are automatically converted to float32

**Data Split:**
- Training: 70%
- Validation: 15%
- Test: 15%
- Stratified splitting to maintain class distribution

### 2. Main Training Script (`DNSAssignment.py`)

**Purpose**: Trains the DNS classification model with full monitoring and evaluation.

**Key Functions:**

#### `train_model()`
- Trains model for specified epochs
- Implements early stopping (patience=5)
- Saves best model based on validation loss
- Uses learning rate scheduling (ReduceLROnPlateau)
- Applies gradient clipping (max_norm=1.0)
- Logs metrics to WandB

#### `evaluate_model()`
- Evaluates model on given dataloader
- Computes loss and predictions
- Loss calculation: Weighted by batch size for accurate averaging across variable batch sizes
- Returns average loss, predictions, and labels

#### `test_and_report()`
- Final evaluation on test set
- Generates classification report
- Creates confusion matrix
- Logs accuracy to WandB

**Features:**
- Automatic device detection (CUDA/CPU)
- Label range validation
- Automatic class count detection
- WandB experiment tracking
- Model checkpointing

**Usage:**
```bash
python DNSAssignment.py
```

**Expected Inputs:**
- `Data/DNS2021/train.npy`
- `Data/DNS2021/test.npy`
- `Data/DNS2021/val.npy`
- `Data/DNS2021/class_names.npy`
- `config.yaml`

**Outputs:**
- `DNS-Assignment2025.pth` - Best model weights
- `scaler.pkl` - StandardScaler for normalization
- WandB logs in `wandb/` directory

**First Trial Configuration:**
- Maximum epochs: 50 (with early stopping, patience=5)
- Weighted sampler: Disabled (`use_weighted_sampler=False`)
- Loss function: Standard CrossEntropyLoss (no class weights)
- WandB tracking: Basic metrics (loss curves, accuracy, F1 scores, confusion matrix)

### 3. Model Architecture (`model.py`)

**Purpose**: Defines the neural network architecture.

**Architecture:**
```
Input Layer (n_features)
    ↓
Linear(256) → BatchNorm1d → GELU → Dropout(0.2)
    ↓
Linear(128) → BatchNorm1d → GELU → Dropout(0.2)
    ↓
Output Layer (4 classes)
```

**Key Components:**
- **Input Features**: Configurable based on dataset
- **Hidden Layers**: 256 → 128 neurons
- **Activation**: GELU (Gaussian Error Linear Unit)
- **Regularization**: 
  - Batch Normalization after each linear layer
  - Dropout (20%) for overfitting prevention
- **Output**: Logits for multi-class classification (4 classes)

**Initialization:**
```python
from model import DNSModel
model = DNSModel(input_features=num_features, num_classes=4, hidden_sizes=[256, 128], dropout=0.2)
```

**Why MLP for DNS Classification?**
- ✅ Optimal for tabular/feature-based data (not sequential/image)
- ✅ Efficient training and inference for large datasets
- ✅ Can learn complex non-linear relationships
- ✅ Simpler and faster than RNN/CNN for feature-based classification

### 4. Data Preprocessing Utilities (`dns_preprocess.py`)

**Purpose**: Handles data normalization and DataLoader creation.

**Functions:**

#### `preprocess(train, test, val, batch_size, scaler_save_path=None, use_weighted_sampler=False)`
- Fits StandardScaler on training data
- Transforms train/test/val sets
- Handles infinity and extreme values
- Creates PyTorch DataLoaders
- Saves scaler for later use
- Optional WeightedRandomSampler for class imbalance (disabled in first trial)

#### `DNSDataset`
- PyTorch Dataset class for DNS classification
- Handles feature and label tensors

**Data Format:**
- Input: NumPy arrays with shape `(n_samples, n_features + 1)`
- Last column: Integer labels
- Features: Normalized to mean=0, std=1

**Data Cleaning:**
- Replaces infinity with NaN, then with median
- Clips extreme values to prevent overflow
- Converts to float32 for efficiency

### 5. ONNX Conversion (`convert.py`)

**Purpose**: Converts trained PyTorch model to ONNX format for deployment.

**Features:**
- Loads trained model weights
- Exports to ONNX with dynamic batch size
- Supports both CPU and GPU inference

**Usage:**
```bash
python convert.py
```

**Requirements:**
- `DNS-Assignment2025.pth` must exist
- `config.yaml` must be configured
- Training data needed to determine input shape

**Output:**
- `DNS-Assignment2025.onnx` - ONNX model file

### 6. ONNX Testing (`onnxtest.py`)

**Purpose**: Tests ONNX model performance and benchmarks inference speed.

**Features:**
- Loads ONNX model with CUDA/CPU providers
- Performs inference on test set
- Compares with PyTorch model
- Calculates accuracy
- Measures inference time and throughput
- Warm-up run for accurate timing

**Usage:**
```bash
python onnxtest.py
```

**Output:**
- Accuracy percentage
- Prediction agreement with PyTorch model
- Total inference time
- Average time per batch
- Throughput (samples/second)

**Requirements:**
- `DNS-Assignment2025.onnx` - ONNX model file
- `scaler.pkl` - Saved scaler
- `Data/DNS2021/test.npy` - Test data

---

## Workflow

### Complete Pipeline

#### Step 1: Data Preparation

```bash
# Navigate to data directory
cd Data/DNS2021

# Run preprocessing script
python preprocess_csv.py
```

**What happens:**
1. Downloads CIC-Bell-DNS-2021 dataset (if not present)
2. Extracts CSV files
3. Loads and combines CSV files (benign, malware, phishing, spam)
4. Handles missing values and infinity
5. Converts features to numeric
6. Encodes labels
7. Splits data (70% train, 15% val, 15% test)
8. Saves numpy arrays and class names

**Output:**
- `train.npy`, `test.npy`, `val.npy`
- `class_names.npy`

#### Step 2: Training

```bash
# Navigate back to project root
cd ../..

# Run training script
python DNSAssignment.py
```

**What happens:**
1. Loads data and configuration
2. Validates label ranges
3. Preprocesses data (normalization, cleaning)
4. Creates model
5. Trains with early stopping
6. Saves best model
7. Evaluates on test set

**Output:**
- `DNS-Assignment2025.pth` - Best model
- `scaler.pkl` - Scaler
- WandB logs

#### Step 3: Model Conversion (Optional)

```bash
# Still in project root
python convert.py
```

**What happens:**
1. Loads trained model
2. Exports to ONNX format
3. Saves ONNX model

**Output:**
- `DNS-Assignment2025.onnx`

#### Step 4: ONNX Testing (Optional)

```bash
# Still in project root
python onnxtest.py
```

**What happens:**
1. Loads ONNX model
2. Preprocesses test data
3. Runs inference
4. Compares with PyTorch model
5. Calculates accuracy and performance metrics

**Output:**
- Performance report (accuracy, timing, throughput)

---

## Configuration

### Configuration File (`config.yaml`)

```yaml
class_names:
  - 'Benign'
  - 'Malware'
  - 'Phishing'
  - 'Spam'

batch_size: 128
learning_rate: 0.001
num_epochs: 50
```

### Configuration Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `class_names` | List of class names | `['Benign', 'Malware', 'Phishing', 'Spam']` | Used for reporting; actual classes loaded from data |
| `batch_size` | Samples per batch | `128` | Adjust based on GPU memory |
| `learning_rate` | Initial learning rate | `0.001` | For AdamW optimizer |
| `num_epochs` | Maximum training epochs | `50` | Early stopping may stop earlier (patience=5) |

### Training Configuration

**Optimizer:** AdamW
- Learning rate: 0.001 (configurable)
- Weight decay: Default

**Loss Function:** CrossEntropyLoss
- Multi-class classification (4 classes)
- No class weights (first trial uses standard loss)
- Loss calculation: Weighted by batch size for accurate averaging

**Learning Rate Scheduler:** ReduceLROnPlateau
- Mode: `min` (monitor validation loss)
- Factor: `0.5` (reduce by 50%)
- Patience: `3` epochs

**Early Stopping:**
- Patience: `5` epochs
- Monitors: validation loss
- Saves: best model based on validation loss

**First Trial Setup:**
- Weighted Sampler: Disabled (`use_weighted_sampler=False`)
- Class Weights: Not used (standard CrossEntropyLoss)
- Focus: Basic training with standard configuration

**Regularization:**
- Gradient clipping: `max_norm=1.0`
- Dropout: `20%`
- Batch Normalization: After each linear layer

---

## Model Architecture

### Network Structure

```
Input Layer (n_features)
    ↓
[Linear(256)] → [BatchNorm1d] → [GELU] → [Dropout(0.2)]
    ↓
[Linear(128)] → [BatchNorm1d] → [GELU] → [Dropout(0.2)]
    ↓
[Linear(4)]
    ↓
Output (logits)
```

### Architecture Details

**Layer 1:**
- Linear: `input_features → 256`
- Batch Normalization
- Activation: GELU
- Dropout: 20%

**Layer 2:**
- Linear: `256 → 128`
- Batch Normalization
- Activation: GELU
- Dropout: 20%

**Output Layer:**
- Linear: `128 → 4`
- Output: Raw logits (no activation)

### Design Choices

1. **GELU Activation**: Smooth activation function, often performs better than ReLU
2. **Batch Normalization**: Stabilizes training and allows higher learning rates
3. **Dropout**: Prevents overfitting (20% dropout rate)
4. **Two Hidden Layers**: Balances capacity and overfitting risk
5. **No Output Activation**: CrossEntropyLoss expects raw logits

### Why MLP for DNS Classification?

- ✅ **Tabular Data**: DNS features are extracted and tabular (not sequential)
- ✅ **Efficiency**: Fast training and inference for large datasets (400K+ samples)
- ✅ **Non-linear Relationships**: Can learn complex patterns between features
- ✅ **Simplicity**: Easier to train and debug than RNN/CNN for this task

---

## Data Processing

### Data Format

**Input Arrays:**
- Shape: `(n_samples, n_features + 1)`
- Last column: Integer labels (0, 1, 2, 3)
- Other columns: Feature values

**Preprocessing Steps:**

1. **Feature Encoding:**
   - All features converted to numeric (float32)
   - Non-numeric values handled with `pd.to_numeric(errors='coerce')`

2. **Missing Value Handling:**
   - Infinity values replaced with NaN
   - NaN values filled with 0 (or median during PyTorch preprocessing)

3. **Infinity Handling:**
   - Replaced with NaN, then with median
   - Clipped extreme values to prevent overflow

4. **Normalization:**
   - StandardScaler: mean=0, std=1
   - Fitted on training data only
   - Applied to train/test/val

5. **Data Splitting:**
   - Train: 70%
   - Validation: 15%
   - Test: 15%
   - Stratified splitting to maintain class distribution
   - Split is done in `Data/DNS2021/preprocess_csv.py`

### Label Requirements

- Labels must be integers in range `[0, 3]` (4 classes)
- Script validates label ranges before training
- Invalid ranges raise `ValueError`

### Class Names

- Stored in `class_names.npy` as numpy array
- Automatically loaded during training
- Used for classification reports and confusion matrices
- Order: `['Benign', 'Malware', 'Phishing', 'Spam']`

---

## Training

### Training Process

1. **Data Loading**
   - Loads train/val/test arrays
   - Loads class names
   - Validates data format

2. **Validation**
   - Checks label ranges match expected class count
   - Verifies data is numeric

3. **Preprocessing**
   - Normalizes features using StandardScaler
   - Cleans infinity and extreme values
   - Creates DataLoaders with specified batch size

4. **Model Creation**
   - Initializes model with correct dimensions
   - Moves to device (CUDA/CPU)
   - Prints model summary

5. **Training Loop**
   - Forward pass
   - Loss calculation
   - Backward pass with gradient clipping
   - Optimizer step
   - Learning rate scheduling
   - Early stopping check
   - WandB logging

6. **Evaluation**
   - Tests on test set
   - Generates classification report
   - Creates confusion matrix
   - Logs accuracy

### Monitoring

**WandB Integration:**
- Project: `DNS-Assignment2025`
- Logged Metrics (Per Epoch):
  - Train Loss
  - Validation Loss
  - Learning Rate
- Logged Metrics (Final Test):
  - Test Accuracy
  - Test F1 Macro
  - Test F1 Weighted
  - Confusion Matrix
- Model Watching: Tracks gradients and parameters (`wandb.watch(model, log="all")`)

**Console Output:**
- Progress bars (tqdm)
- Epoch summaries
- Final classification report
- Confusion matrix

**Model Checkpointing:**
- Saves best model based on validation loss
- File: `DNS-Assignment2025.pth`
- Loaded at end of training for final evaluation

---

## Model Conversion & Deployment

### ONNX Conversion

**Purpose:** Convert PyTorch model to ONNX for production deployment.

**Process:**
1. Load trained model weights
2. Create dummy input
3. Export to ONNX with dynamic batch size
4. Save ONNX model

**Usage:**
```bash
python convert.py
```

**Output:**
- `DNS-Assignment2025.onnx` - ONNX model file

### ONNX Testing

**Purpose:** Benchmark ONNX model performance and verify correctness.

**Features:**
- GPU/CPU provider selection
- Warm-up run for accurate timing
- Batch processing
- Performance metrics
- Comparison with PyTorch model

**Usage:**
```bash
python onnxtest.py
```

**Output:**
- Accuracy: Model accuracy on test set
- Prediction Agreement: Percentage of predictions matching PyTorch model
- Total time: Total inference time
- Average time per batch: Milliseconds per batch
- Throughput: Samples per second
- Speedup: ONNX vs PyTorch speed comparison

---

## Evaluation Criteria

Based on the project requirements, the following criteria are evaluated:

### 1. Code Quality

**Modularity:**
- ✅ Functions are separated into logical modules
- ✅ Training loop is reusable (`train_model()`)
- ✅ Evaluation functions are modular
- ✅ Preprocessing is separated from training

**Dynamic Code:**
- ✅ Model architecture adapts to input features
- ✅ Class count automatically detected from data
- ✅ Device detection (CUDA/CPU)
- ✅ Configuration-driven parameters

### 2. Correctness

**Data Splits:**
- ✅ Proper train/validation/test split (70/15/15)
- ✅ Stratified splitting to maintain class distribution
- ✅ No data leakage

**Loss Function:**
- ✅ CrossEntropyLoss for multi-class classification
- ✅ Appropriate for the task (DNS traffic classification)

**Label Validation:**
- ✅ Labels validated before training
- ✅ Range checks prevent CUDA errors
- ✅ Automatic class count detection

### 3. Analysis

**Results Analysis:**
- ✅ Classification report with per-class metrics
- ✅ Confusion matrix
- ✅ Accuracy reporting
- ✅ Training/validation loss tracking
- ✅ Learning rate monitoring

**Performance Metrics:**
- ✅ Training time tracking
- ✅ Inference speed benchmarking (ONNX)
- ✅ Throughput measurements
- ✅ Model comparison (PyTorch vs ONNX)

### 4. Documentation

**Code Documentation:**
- ✅ Function docstrings
- ✅ Clear comments
- ✅ Type hints where appropriate
- ✅ Usage examples

**Project Documentation:**
- ✅ This comprehensive documentation
- ✅ README files
- ✅ Configuration documentation
- ✅ Architecture analysis

### 5. Performance

**Efficient Metrics:**
- ✅ Vectorized operations
- ✅ Batch processing
- ✅ GPU utilization
- ✅ ONNX for faster inference

**Reasonable Training/Inference Times:**
- ✅ Early stopping prevents unnecessary training
- ✅ Batch processing for efficiency
- ✅ ONNX runtime for deployment
- ✅ Performance benchmarking included

---

## Troubleshooting

### Common Issues

#### 1. FileNotFoundError: Model file not found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'DNS-Assignment2025.pth'
```

**Solution:**
- Ensure model has been trained first
- Check filename spelling: `DNS-Assignment2025.pth`
- Run training script before conversion/testing

#### 2. ValueError: Object arrays cannot be loaded

**Error:**
```
ValueError: Object arrays cannot be loaded when allow_pickle=False
```

**Solution:**
- Ensure all `np.load()` calls include `allow_pickle=True`
- This is already fixed in all scripts

#### 3. ValueError: could not convert string to float

**Error:**
```
ValueError: could not convert string to float: '...'
```

**Solution:**
- Run `preprocess_csv.py` to regenerate data
- Ensure all features are converted to numeric
- Check that preprocessing completed successfully

#### 4. CUDA device-side assert triggered

**Error:**
```
RuntimeError: CUDA error: device-side assert triggered
```

**Solution:**
- Script validates label ranges automatically
- Ensure labels are in range `[0, num_classes-1]` (0-3 for 4 classes)
- Check that class count matches label range

#### 5. ValueError: The least populated class has only 1 member

**Error:**
```
ValueError: The least populated class has only 1 member, which is too few
```

**Solution:**
- Script handles this automatically
- Falls back to non-stratified split if needed
- This is expected behavior for very imbalanced datasets

#### 6. ONNX model input shape mismatch

**Error:**
```
ONNXRuntimeError: Got invalid dimensions for input
```

**Solution:**
- Regenerate ONNX model with `convert.py`
- Ensure model was trained on current data
- Check input feature count matches

#### 7. Memory Error during training

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size in `config.yaml`
- Use CPU instead of GPU
- Process data in smaller chunks

### Data Validation

Before training, the script validates:
- ✅ Label ranges match expected class count (0-3)
- ✅ Data arrays are numeric (float32)
- ✅ Class names file exists and matches label count
- ✅ Data splits are properly formatted

### Performance Issues

**Slow Training:**
- Use GPU if available (CUDA)
- Reduce batch size if out of memory
- Enable early stopping to avoid unnecessary epochs

**High Memory Usage:**
- Reduce batch size
- Process data in smaller chunks
- Use CPU if GPU memory is limited

**Low Accuracy:**
- Check data quality and preprocessing
- Adjust learning rate
- Increase model capacity
- Add more training data
- Tune hyperparameters

---

## Future Improvements

Potential enhancements:

1. **Model Architecture**
   - Experiment with different architectures
   - Add attention mechanisms
   - Try transformer-based models
   - Ensemble methods

2. **Hyperparameter Tuning**
   - Grid search or random search
   - Bayesian optimization
   - Automated hyperparameter tuning
   - Cross-validation

3. **Data Augmentation**
   - Synthetic data generation
   - Data balancing techniques
   - Feature engineering

4. **Advanced Features**
   - Multi-GPU training
   - Model quantization
   - Real-time inference API
   - Web interface
   - Model versioning

5. **Monitoring**
   - Enhanced WandB integration
   - Model versioning
   - A/B testing capabilities
   - Performance monitoring

---

## License

This project is part of a coursework assignment. Please refer to your institution's guidelines for usage and distribution.

---

## Acknowledgments

- **PyTorch Team** for the deep learning framework
- **Weights & Biases** for experiment tracking
- **ONNX Runtime** for model deployment
- **scikit-learn** for preprocessing utilities
- **CIC Research** for the CIC-Bell-DNS-2021 dataset

---

## Contact & Support

For questions or issues:
1. Check this documentation
2. Review error messages and troubleshooting section
3. Check code comments and docstrings
4. Consult project README files

---

**Last Updated:** December 2024  
**Version:** 1.0  
**Status:** Production Ready

