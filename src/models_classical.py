import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report, accuracy_score, f1_score

from utils import SEED, set_seed, load_data, encode_labels, create_train_val_split
from preprocessing import preprocess_for_classical, apply_text_combination, apply_classical_preprocessing
from augmentation import augment_minority_classes_synonym


def build_mnb_baseline_vectorizer():
    """
    Builds the baseline TF-IDF vectorizer:
    Unigrams only, max 10,000 features, min_df=2, sublinear TF scaling.
    """

    return TfidfVectorizer(
        ngram_range=(1, 1),
        max_features=10000,
        min_df=2,
        sublinear_tf=True,
    )


def build_strong_tfidf_union():
    """
    Builds a strong TF-IDF feature union combining:
    1. Word n-grams (1-2) up to 50,000 features
    2. Character n-grams (3-5) within word boundaries up to 50,000 features
    """

    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        sublinear_tf=True,
        analyzer="word",
    )
    char_tfidf = TfidfVectorizer(
        ngram_range=(3, 5),
        max_features=50000,
        min_df=2,
        sublinear_tf=True,
        analyzer="char_wb",
    )
    return FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf),
    ])


def train_evaluate_mnb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    label_encoder,
    alpha: float = 1.0,
):
    """
    Trains and evaluates Phase 4.1: Model A (Multinomial Naive Bayes Weak Baseline).
    Note: Following the project description baseline, this only uses the 'description' field.
    """

    # 1. Preprocess description text
    print("Preprocessing 'description' column for classical baseline...")
    train_desc_clean = X_train["description"].apply(preprocess_for_classical)
    val_desc_clean = X_val["description"].apply(preprocess_for_classical)

    # 2. Vectorize (Fit ONLY on training split!)
    vectorizer = build_mnb_baseline_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(train_desc_clean)
    X_val_tfidf = vectorizer.transform(val_desc_clean)

    # 3. Train Model
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_train_tfidf, y_train)

    # 4. Predict & Evaluate on Validation Set
    y_pred = mnb.predict(X_val_tfidf)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    weighted_f1 = f1_score(y_val, y_pred, average="weighted")

    print("\n" + "=" * 60)
    print(" Phase 4.1: Multinomial Naive Bayes Baseline (Model A)")
    print("=" * 60)
    print(f"Validation Accuracy:  {acc:.4f} (~{acc*100:.2f}%)")
    print(f"Validation Macro F1:  {macro_f1:.4f}")
    print(f"Validation Weighted F1: {weighted_f1:.4f}\n")
    print("Detailed Classification Report:")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

    return {
        "model": mnb,
        "vectorizer": vectorizer,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "y_pred": y_pred,
    }


def train_evaluate_linearsvc(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    label_encoder,
    text_col: str = "text_classical",
    C: float = 1.0,
    model_name: str = "Model B: LinearSVC (Strong Classical)",
):
    """
    Trains and evaluates LinearSVC on combined word+character TF-IDF features.
    Uses class_weight='balanced' to address class imbalance.
    """

    feature_union = build_strong_tfidf_union()

    # Fit feature union strictly on X_train, then transform both train and val
    X_train_vec = feature_union.fit_transform(X_train[text_col])
    X_val_vec = feature_union.transform(X_val[text_col])

    svc = LinearSVC(
        C=C,
        max_iter=10000,
        random_state=SEED,
        class_weight="balanced",
    )
    svc.fit(X_train_vec, y_train)

    y_pred = svc.predict(X_val_vec)
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    weighted_f1 = f1_score(y_val, y_pred, average="weighted")

    print("\n" + "=" * 60)
    print(f" Phase 4.2: {model_name}")
    print("=" * 60)
    print(f"Validation Accuracy:    {acc:.4f} (~{acc*100:.2f}%)")
    print(f"Validation Macro F1:    {macro_f1:.4f}")
    print(f"Validation Weighted F1: {weighted_f1:.4f}\n")
    print("Detailed Classification Report:")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

    return {
        "model": svc,
        "feature_union": feature_union,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    set_seed(SEED)

    # 1. Load, encode, combine text, preprocess, and split
    train_df, test_df = load_data()
    train_df, label_encoder = encode_labels(train_df)
    train_df = apply_text_combination(train_df)
    train_df = apply_classical_preprocessing(train_df, source_col="text_all")
    X_train, X_val, y_train, y_val = create_train_val_split(train_df)

    # 2. Phase 4.1: Model A Baseline (MultinomialNB on description only)
    results_mnb = train_evaluate_mnb(X_train, y_train, X_val, y_val, label_encoder)

    # 3. Phase 4.2: Model B Strong Classical (LinearSVC on all fields with word+char TF-IDF)
    results_svc = train_evaluate_linearsvc(
        X_train, y_train, X_val, y_val, label_encoder,
        text_col="text_classical",
        model_name="Model B: LinearSVC (No Augmentation)"
    )

    # 4. Phase 4.2.1: Ablation with Data Augmentation
    print("\nAugmenting training set with minority synonym replacement...")
    X_train_aug = augment_minority_classes_synonym(X_train, label_col="label", text_col="text_all", n_augments=2)
    results_svc_aug = train_evaluate_linearsvc(
        X_train_aug, X_train_aug["label"], X_val, y_val, label_encoder,
        text_col="text_classical",
        model_name="Model B: LinearSVC (With Synonym Augmentation)"
    )

    # 5. Phase 4.2.1: Ablation - LinearSVC with Synonym Augmentation
    print("\nApplying synonym augmentation to minority classes for ablation...")
    X_train_aug = augment_minority_classes_synonym(X_train, label_col="label", text_col="text_all", n_augments=2)
    results_svc_aug = train_evaluate_linearsvc(
        X_train_aug, X_train_aug["label"], X_val, y_val, label_encoder,
        text_col="text_classical",
        model_name="Model B: LinearSVC (With Synonym Augmentation)"
    )
