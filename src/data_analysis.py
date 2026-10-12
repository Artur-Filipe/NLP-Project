import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, set_seed, SEED

def plot_class_distribution(train_df, output_dir="figures"):
    """Phase 1.1: Compute distribution and save bar plot."""
    os.makedirs(output_dir, exist_ok=True)

    class_counts = train_df["medical_specialty"].value_counts()
    print("== Class Counts ==")
    print(class_counts)
    print("\n== Class Percentages ==")
    print((class_counts / len(train_df) * 100).round(2))

    # Filter to top 12 specialties to avoid corrupted/multiline text labels from anomalous rows
    class_counts = class_counts.head(12)

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    class_counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("Medical Specialty")
    ax.set_title("Class Distribution in Training Set")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {out_path}")


def analyze_text_lengths(train_df, output_dir="figures"):
    """Phase 1.2: Compute word lengths across columns and boxplot per specialty."""
    
    os.makedirs(output_dir, exist_ok=True)

    # 1. Compute word count for each text column
    text_cols = ["description", "sample_name", "transcription", "keywords"]
    for col in text_cols:
        train_df[f"{col}_len"] = train_df[col].apply(lambda x: len(str(x).split()))

    # 2. Print summary statistics table for the paper
    len_cols = [f"{col}_len" for col in text_cols]
    print("\n== Text Length Summary Statistics (Word Counts) ==")
    print(train_df[len_cols].describe().round(1))

    # 3. Filter to top 12 specialties for a clean box plot
    top_specialties = train_df["medical_specialty"].value_counts().head(12).index
    clean_df = train_df[train_df["medical_specialty"].isin(top_specialties)].copy()

    # 4. Box plot of transcription length per specialty
    fig, ax = plt.subplots(figsize=(12, 6))
    clean_df.boxplot(column="transcription_len", by="medical_specialty", ax=ax, rot=45)
    plt.suptitle("")  # Remove default pandas supertitle
    ax.set_title("Transcription Length by Specialty")
    ax.set_xlabel("Medical Specialty")
    ax.set_ylabel("Word Count")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "transcription_length_by_specialty.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {out_path}")


def analyze_keyword_leakage(train_df):
    """Phase 1.3: Check how often the keywords field contains the specialty name."""

    # Filter to top 12 specialties to exclude corrupted rows
    top_specialties = train_df["medical_specialty"].value_counts().head(12).index
    clean_df = train_df[train_df["medical_specialty"].isin(top_specialties)].copy()

    def keyword_contains_specialty(row):
        specialty_words = row["medical_specialty"].lower().replace("-", " ").split()
        keywords_lower = str(row["keywords"]).lower().replace("-", " ")
        
        # Match significant words (> 3 chars) to avoid false positives with short stop words
        return any(w in keywords_lower for w in specialty_words if len(w) > 3)

    clean_df["keyword_leakage"] = clean_df.apply(keyword_contains_specialty, axis=1)
    leakage_rate = clean_df["keyword_leakage"].mean()

    print("\n== Phase 1.3: Keyword Leakage Analysis ==")
    print(f"Overall Keyword Leakage Rate: {leakage_rate:.1%}")
    print("\nLeakage Rate per Specialty (%):")
    per_specialty = (clean_df.groupby("medical_specialty")["keyword_leakage"].mean().sort_values(ascending=False) * 100).round(1)
    print(per_specialty)


def check_missing_and_noise(train_df):
    """Phase 1.4: Check for missing fields, very short transcriptions, and CSV noise."""

    text_cols = ["description", "sample_name", "transcription", "keywords"]

    print("\n== Phase 1.4: Missing Values and Annotation Noise ==")
    
    # 1. Check for empty string fields (since load_data fills NaNs with "")
    print("Empty / Blank fields per column:")
    for col in text_cols:
        empty_count = (train_df[col].astype(str).str.strip() == "").sum()
        print(f"  - {col}: {empty_count} ({empty_count / len(train_df):.1%})")

    # 2. Potential mislabels: extremely short transcriptions (< 10 words)
    if "transcription_len" not in train_df.columns:
        train_df["transcription_len"] = train_df["transcription"].apply(lambda x: len(str(x).split()))
    
    short_trans = train_df[train_df["transcription_len"] < 10]
    print(f"\nRecords with < 10 words in transcription: {len(short_trans)}")
    if len(short_trans) > 0:
        print("Sample short transcriptions:")
        print(short_trans[["medical_specialty", "description", "transcription"]].head(3))

    # 3. Annotation/parsing noise in target labels
    top_specialties = train_df["medical_specialty"].value_counts().head(12).index
    corrupted_rows = train_df[~train_df["medical_specialty"].isin(top_specialties)]
    print(f"\nCorrupted/misaligned specialty rows: {len(corrupted_rows)}")
    if len(corrupted_rows) > 0:
        print("Indices of corrupted rows:", corrupted_rows.index.tolist())



if __name__ == "__main__":
    set_seed(SEED)
    train_df, _ = load_data()
    plot_class_distribution(train_df)
    analyze_text_lengths(train_df)
    analyze_keyword_leakage(train_df)
    check_missing_and_noise(train_df)
