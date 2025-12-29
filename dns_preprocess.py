"""
PyTorch-compatible data preprocessing for DNS classification.
Converts preprocessed data to PyTorch DataLoaders.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
import pickle


class DNSDataset(Dataset):
    """PyTorch Dataset for DNS classification."""

    def __init__(self, features, labels):
        if isinstance(features, pd.DataFrame):
            self.features = torch.FloatTensor(features.values)
        else:
            self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def preprocess(
    train_data,
    test_data,
    val_data,
    batch_size=128,
    scaler_save_path="scaler.pkl",
    use_weighted_sampler=False,
):
    """
    Preprocess data and create PyTorch DataLoaders.

    Args:
        train_data (np.ndarray): shape (n_samples, n_features + 1), last col is labels
        test_data  (np.ndarray): shape (n_samples, n_features + 1)
        val_data   (np.ndarray): shape (n_samples, n_features + 1)
        batch_size (int)
        scaler_save_path (str)
        use_weighted_sampler (bool): if True, balance classes during training via sampler

    Returns:
        (train_loader, test_loader, val_loader)
    """
    print("Preprocessing data for PyTorch...")

    # Separate features and labels
    X_train = train_data[:, :-1].astype(np.float64)
    y_train = train_data[:, -1].astype(np.int64)

    X_val = val_data[:, :-1].astype(np.float64)
    y_val = val_data[:, -1].astype(np.int64)

    X_test = test_data[:, :-1].astype(np.float64)
    y_test = test_data[:, -1].astype(np.int64)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")

    def clean_data(X: np.ndarray) -> np.ndarray:
        """Replace inf -> nan, fill nan with median, clip extremes, return float32."""
        X = np.where(np.isinf(X), np.nan, X)

        # fill NaN with per-column median
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            if np.any(np.isnan(col)):
                med = np.nanmedian(col)
                if np.isnan(med) or not np.isfinite(med):
                    med = 0.0
                X[:, col_idx] = np.where(np.isnan(col), med, col)

        # clip extreme values
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            try:
                abs_col = np.abs(col)
                if np.any(abs_col > 0):
                    p99 = np.percentile(abs_col[abs_col > 0], 99)
                else:
                    p99 = 1.0
                max_val = min(max(p99 * 10, 1e6), 1e6)
                X[:, col_idx] = np.clip(col, -max_val, max_val)
            except Exception:
                X[:, col_idx] = np.clip(col, -1e6, 1e6)

        return X.astype(np.float32)

    print("Cleaning data (handling infinity and extreme values)...")
    X_train = clean_data(X_train)
    X_val = clean_data(X_val)
    X_test = clean_data(X_test)

    # final safety
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with open(scaler_save_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_save_path}")

    train_dataset = DNSDataset(X_train_scaled, y_train)
    val_dataset = DNSDataset(X_val_scaled, y_val)
    test_dataset = DNSDataset(X_test_scaled, y_test)

    # Weighted sampler (training only)
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        print("Using WeightedRandomSampler for training (class balancing).")
        class_counts = np.bincount(y_train)
        class_counts[class_counts == 0] = 1
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        sample_weights = torch.DoubleTensor(sample_weights)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False  # shuffle cannot be True when sampler is set

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print("DataLoaders created successfully!")
    return train_loader, test_loader, val_loader
