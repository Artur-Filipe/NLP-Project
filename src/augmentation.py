from sklearn.utils import resample

import nltk
import pandas as pd
import nlpaug.augmenter.word as naw

from utils import SEED, set_seed, load_data, encode_labels, create_train_val_split
from preprocessing import apply_text_combination, preprocess_for_classical, apply_classical_preprocessing

# Ensure NLTK resources required by nlpaug are downloaded
for resource in ["wordnet", "omw-1.4", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(resource, quiet=True)


def get_synonym_augmenter(aug_p=0.1, aug_min=1, aug_max=5):
    """
    Initializes a WordNet-based Synonym Augmenter.
    Preserves medical text while injecting linguistic variation.
    """

    return naw.SynonymAug(
        aug_src="wordnet",
        aug_min=aug_min,
        aug_max=aug_max,
        aug_p=aug_p,
    )


def augment_text_synonym(text: str, augmenter=None, n_augments=1) -> list:
    """Generate n_augments augmented versions of a single text string."""

    if augmenter is None:
        augmenter = get_synonym_augmenter()

    if not str(text).strip():
        return [text] * n_augments

    augmented = []
    for _ in range(n_augments):
        aug_text = augmenter.augment(text)
        if isinstance(aug_text, list):
            augmented.append(aug_text[0])
        else:
            augmented.append(aug_text)
    return augmented


def augment_minority_classes_synonym(
    X_train: pd.DataFrame,
    label_col: str = "label",
    text_col: str = "text_all",
    n_augments: int = 2,            #2 augmented copies per minority sample
    threshold: str = "median",
) -> pd.DataFrame:
    """
    Phase 3.2 Strategy: Apply synonym replacement ONLY to minority classes in the training set.
    NEVER apply to validation or test data!
    """

    class_counts = X_train[label_col].value_counts()
    cutoff = class_counts.median() if threshold == "median" else threshold
    minority_classes = class_counts[class_counts < cutoff].index.tolist()
    augmenter = get_synonym_augmenter()
    augmented_rows = []
    minority_df = X_train[X_train[label_col].isin(minority_classes)]
    for _, row in minority_df.iterrows():
        aug_texts = augment_text_synonym(row[text_col], augmenter=augmenter, n_augments=n_augments)
        for aug_text in aug_texts:
            new_row = row.copy()
            new_row[text_col] = aug_text

            # Recompute classical stemmed text if that column exists
            if "text_classical" in row:
                new_row["text_classical"] = preprocess_for_classical(aug_text)
            augmented_rows.append(new_row)

    if augmented_rows:
        return pd.concat([X_train, pd.DataFrame(augmented_rows)], ignore_index=True)
    return X_train.copy()


def oversample_minority_classes(df: pd.DataFrame, target_col: str = "label", target_count: int = None, seed: int = SEED) -> pd.DataFrame:
    """
    Phase 3.1 Technique B: Directly duplicate samples from minority classes 
    to match the majority class count (or a custom target_count).

    NOTE: Must ONLY be applied to X_train (never validation or test sets).
    """

    class_counts = df[target_col].value_counts()
    if target_count is None:
        target_count = class_counts.max()

    frames = []
    for label, count in class_counts.items():
        class_df = df[df[target_col] == label]
        if count < target_count:
            oversampled = resample(
                class_df,
                replace=True,
                n_samples=target_count,
                random_state=seed,
            )
            frames.append(oversampled)
        else:
            frames.append(class_df)

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    set_seed(SEED)

    # 1. Quick sanity check on single sentence

    sample_text = "Patient presents with persistent cough and shortness of breath."
    aug = get_synonym_augmenter(aug_p=0.2)
    print("Original text:", sample_text)
    print("Augmented text:", augment_text_synonym(sample_text, augmenter=aug, n_augments=1)[0])

    # 2. Pipeline setup: Load, encode, combine text, preprocess, and split (80/20)

    print("\nLoading data and creating 80/20 train/val split...")
    train_df, test_df = load_data()
    train_df, _ = encode_labels(train_df)
    train_df = apply_text_combination(train_df)
    train_df = apply_classical_preprocessing(train_df, source_col="text_all")
    X_train, X_val, y_train, y_val = create_train_val_split(train_df)

    # 3. Phase 3.2: Apply Synonym Augmentation to Training Set ONLY

    print("\n== Phase 3.2: Synonym Augmentation (Minority Classes Only) ==")
    print(f"Original train size: {len(X_train)}")

    # We test on a small subset or full set (Note: full text augmentation can take ~1-2 min)
    X_train_syn_aug = augment_minority_classes_synonym(
        X_train,
        label_col="label",
        text_col="text_all",
        n_augments=2
    )
    print(f"After synonym augmentation: {len(X_train_syn_aug)}")
    print("New class distribution:\n", X_train_syn_aug["label"].value_counts().sort_index())

    # 4. Technique B Demonstration: Oversampling
    print("\n== Technique B: Random Oversampling ==")
    X_train_oversampled = oversample_minority_classes(X_train, target_col="label")
    print(f"After random oversampling: {len(X_train_oversampled)}")
    print("Oversampled class distribution:\n", X_train_oversampled["label"].value_counts().sort_index())