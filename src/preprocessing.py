import os
import re
import pandas as pd
from nltk.stem import PorterStemmer
from utils import load_data, set_seed, SEED

porter_stemmer = PorterStemmer()


def combine_text_fields(row, include_keywords=True):
    """Combine all text fields into a single structured string with field markers."""

    parts = [
        f"Title: {row['sample_name']}",
        f"Summary: {row['description']}",
        f"Clinical Note: {row['transcription']}",
    ]
    if include_keywords and str(row.get("keywords", "")).strip():
        parts.append(f"Keywords: {row['keywords']}")
    return " ".join(parts)


def preprocess_for_classical(text: str) -> str:
    """
    Preprocess text for classical ML: lowercases, strips non-alphabet characters,
    and applies Porter stemming.
    NOTE: Stop words are deliberately NOT removed to preserve clinical negation semantics
    (e.g., 'no fracture' vs 'fracture').
    """

    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)       # Keep only alphabetic characters and spaces
    text = re.sub(r"\s+", " ", text).strip()     # Collapse extra whitespace
    words = text.split()
    stemmed = [porter_stemmer.stem(w) for w in words]
    return " ".join(stemmed)


def apply_text_combination(df, is_train=True):
    """
    Adds 'text_all' (with keywords) and 'text_no_kw' (without keywords)
    to the dataframe for ablation studies.
    """

    df = df.copy()
    df["text_all"] = df.apply(lambda r: combine_text_fields(r, include_keywords=True), axis=1)
    df["text_no_kw"] = df.apply(lambda r: combine_text_fields(r, include_keywords=False), axis=1)
    return df


def apply_classical_preprocessing(df: pd.DataFrame, source_col: str = "text_all") -> pd.DataFrame:
    """Applies classical preprocessing to a specified text column."""

    df = df.copy()
    df["text_classical"] = df[source_col].apply(preprocess_for_classical)
    return df


if __name__ == "__main__":
    set_seed(SEED)
    train_df, test_df = load_data()


    # Phase 2.1: Combine text fields
    train_df = apply_text_combination(train_df)
    test_df = apply_text_combination(test_df)

    # Phase 2.2: Classical preprocessing (stemming + cleaning)
    print("Applying classical preprocessing (this takes ~10-15 seconds)...")
    train_df = apply_classical_preprocessing(train_df, source_col="text_all")
    test_df = apply_classical_preprocessing(test_df, source_col="text_all")

    print("== Phase 2.1: Text Combination Preview ==")
    print(f"Sample combined text (all fields):\n{train_df['text_all'].iloc[0][:300]}...\n")
    print(f"Sample combined text (without keywords):\n{train_df['text_no_kw'].iloc[0][:300]}...")

    print("\n== Phase 2.2: Classical Text Preview ==")
    print("Raw Combined Text:")
    print(f"  {train_df['text_all'].iloc[0][:150]}...\n")
    print("Preprocessed Classical Text (Lowercased + Stemmed):")
    print(f"  {train_df['text_classical'].iloc[0][:150]}...\n")