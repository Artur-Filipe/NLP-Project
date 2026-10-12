import os
import random
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

# Global default seed
SEED = 42

def set_seed(seed: int = SEED):
    """Set random seeds across random, numpy, and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Optional: ensure deterministic operations for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def load_data(data_dir="StudentsPack"):
    """Load train and test datasets, handle missing values."""

    # ── Phase 0.2: Training set ──
    train_path = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_path, sep=";")
    # Columns: medical_specialty, description, sample_name, transcription, keywords

    # ── Test set (NO HEADER!) ──
    test_cols = ["description", "sample_name", "transcription", "keywords"]
    test_path = os.path.join(data_dir, "test_no_labels.csv")
    test_df = pd.read_csv(
        test_path,
        sep=";",
        header=None,
        names=test_cols,
    )

    # Handle missing text fields (some records have NaN in transcription or description)
    for col in test_cols:
        train_df[col] = train_df[col].fillna("")
        test_df[col] = test_df[col].fillna("")
    
    return train_df, test_df

def encode_labels(train_df):
    """
    Fit a LabelEncoder on medical_specialty and add a 'label' column.
    Returns:
        train_df (pd.DataFrame): DataFrame containing the new 'label' column.
        label_encoder (LabelEncoder): Fitted encoder for future inverse_transform.
    """
    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df["medical_specialty"])
    return train_df, label_encoder


if __name__ == "__main__":
    # Test script execution
    # Ensure reproducible testing

    set_seed(SEED)

    train_df, test_df = load_data()
    # Verify counts
    print(f"Train samples: {len(train_df)}")    # expect ~2669
    print(f"Test samples:  {len(test_df)}")     # expect 397

    train_df, le = encode_labels(train_df)
    print(f"Number of classes: {len(le.classes_)}")
    print("Classes:", le.classes_)