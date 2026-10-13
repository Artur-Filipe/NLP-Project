import os
import random
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

# Global default seed
SEED = 42

VALID_SPECIALTIES = [
    "Cardiovascular-Pulmonary",
    "Dermatology",
    "Gastroenterology",
    "General Medicine",
    "Neurology",
    "Neurosurgery",
    "Obstetrics-Gynecology",
    "Ophthalmology",
    "Orthopedic",
    "Psychiatry-Psychology",
    "Radiology",
    "Surgery",
]


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


def clean_training_labels(train_df, valid_labels=VALID_SPECIALTIES):
    """
    Remove malformed rows whose medical_specialty is not one of the 12 official classes.

    The raw train.csv contains a few CSV-misaligned rows where long transcription
    fragments appear in the label column. These cannot be valid model targets.
    """

    train_df = train_df.copy()
    valid_labels = set(valid_labels)

    invalid_mask = ~train_df["medical_specialty"].isin(valid_labels)
    invalid_count = invalid_mask.sum()

    if invalid_count:
        print(f"Removing {invalid_count} malformed training rows with invalid labels:")
        for idx, label in train_df.loc[invalid_mask, "medical_specialty"].items():
            print(f"  - row {idx}: {str(label)[:100]}...")

    return train_df.loc[~invalid_mask].reset_index(drop=True)


def load_data(data_dir="StudentsPack"):
    """Load train and test datasets, handle missing values, and clean malformed labels."""

    # Phase 0.2: Training set
    train_path = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_path, sep=";")
    # Columns: medical_specialty, description, sample_name, transcription, keywords

    # Test set (NO HEADER!)
    test_cols = ["description", "sample_name", "transcription", "keywords"]
    test_path = os.path.join(data_dir, "test_no_labels.csv")
    test_df = pd.read_csv(
        test_path,
        sep=";",
        header=None,
        names=test_cols,
    )

    # Handle missing text fields
    for col in test_cols:
        train_df[col] = train_df[col].fillna("")
        test_df[col] = test_df[col].fillna("")

    train_df = clean_training_labels(train_df)

    return train_df, test_df




def encode_labels(train_df):
    """
    Fit a LabelEncoder on medical_specialty and add a 'label' column.
    Returns:
        train_df (pd.DataFrame): DataFrame containing the new 'label' column.
        label_encoder (LabelEncoder): Fitted encoder for future inverse_transform.
    """

    unexpected = sorted(set(train_df["medical_specialty"]) - set(VALID_SPECIALTIES))
    if unexpected:
        raise ValueError(f"Unexpected labels found before encoding: {unexpected}")

    train_df = train_df.copy()
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