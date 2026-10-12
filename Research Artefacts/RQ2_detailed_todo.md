# RQ2_detailed_todo

# RQ2: Classical ML vs. Transformers on Clinical Text — Complete Implementation Guide

> **Research Question:** *“How does classical machine learning with handcrafted features compare to fine-tuned pre-trained transformers for medical specialty classification of clinical notes, and under what conditions does each approach excel?”*
> 

> **Deadline:** October 16, 2026, 23:59 via Fénix
> 

---

## Grading Quick Reference

| Section | Points | Key Deliverable |
| --- | --- | --- |
| Introduction (RQ) | 1.5 | Clear RQ, story of the paper |
| Data Analysis | 1.5 | ≥2 concrete observations (generic = 0) |
| Data Augmentation | 2.0 | **Mandatory** — strategy + rationale |
| Models (desc + creativity) | 4.0 | ≥3 models, replicable, creative |
| Experimental Setup | 1.0 | Splits, metrics, hyperparams |
| Results | 1.5 | Per-class, confusion matrix |
| Discussion | 3.0 | ≥3 concrete misclassified examples + tie to RQ |
| Future Work | 0.5 | Framed around the RQ |
| References + LLM declaration | 1.0 | Cite everything, declare LLM use |
| Beat baseline (Acc > 49%) | 2.0 | Automatic evaluation |
| No class F1 < 25% | 2.0 | Automatic evaluation |
| **Total** | **20.0** |  |

---

---

# Phase 0 — Project Setup

> **Why:** A clean, reproducible environment is not optional — the course can ask you to re-run your code in an oral defense. If you cannot replicate your results, you get **0 points** for the automatic evaluation (4 pts).
> 

## 0.1 Environment

```bash
# Create a conda environment (or venv)
conda create -n nlp-proj python=3.10 -y
conda activate nlp-proj

# Core libraries
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud nltk

# Deep learning / Transformers
pip install torch transformers datasets evaluate accelerate

# Data augmentation
pip install nlpaug  # for synonym replacement, back-translation, etc.

# Optional but useful
pip install textblob spacy
python -m spacy download en_core_web_sm
```

> Save exact library versions for the paper:
> 
> 
> ```bash
> pip freeze > requirements.txt
> ```
> 

## 0.2 Data Loading

> **Why:** The test file has **NO HEADER** — if you load it naively with pandas, it will consume the first test sample as column names, losing 1 sample and misaligning all predictions. This alone could cost you 2.0 points if `results.txt` has the wrong number of lines.
> 

```python
import pandas as pd
import numpy as np

# ── Training set ──
train_df = pd.read_csv("StudentsPack/train.csv", sep=";")
# Columns: medical_specialty, description, sample_name, transcription, keywords

# ── Test set (NO HEADER!) ──
test_cols = ["description", "sample_name", "transcription", "keywords"]
test_df = pd.read_csv(
    "StudentsPack/test_no_labels.csv",
    sep=";",
    header=None,
    names=test_cols,
)

# Verify counts
print(f"Train samples:{len(train_df)}")   # expect ~2669
print(f"Test  samples:{len(test_df)}")     # expect 397

# Handle missing text fields (some records have NaN in transcription or description)
for col in ["description", "sample_name", "transcription", "keywords"]:
    train_df[col] = train_df[col].fillna("")
    test_df[col] = test_df[col].fillna("")
```

## 0.3 Label Encoding

> **Why:** Scikit-learn classifiers and Hugging Face models need integer labels, not strings.
> 

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["medical_specialty"])
# label_encoder.classes_ gives you the ordered list of 12 specialty names
# label_encoder.inverse_transform([0,1,...]) converts back

n_classes = len(label_encoder.classes_)
print(f"Number of classes:{n_classes}")  # 12
print(label_encoder.classes_)
```

## 0.4 Set Random Seeds

> **Why:** Reproducibility (evaluated under Replicability, 1.5 pts).
> 

```python
import random, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

---

# Phase 1 — Data Analysis *(Section 2.1 — 1.5 pts)*

> **Why this phase exists:**
The template says: *“Describe the dataset (class distribution and relevant statistics); provide at least 2 concrete observations about patterns or anomalies you found by looking at the data. Generic descriptions score zero.”*
> 
> 
> This means you MUST actually open the data, read specific records, and find real patterns — not just say “the dataset is imbalanced”. Every claim needs concrete numbers or examples.
> 

## 1.1 Class Distribution Analysis

> **What to do:** Compute and visualize exact counts per specialty.
**Why:** This is the foundation of the Data Analysis section. The 35:1 ratio (Surgery vs Dermatology) is a key property that drives augmentation decisions, model evaluation choices, and error patterns.
> 

```python
import matplotlib.pyplot as plt
import seaborn as sns

class_counts = train_df["medical_specialty"].value_counts()
print(class_counts)

# ── Bar plot ──
fig, ax = plt.subplots(figsize=(10, 5))
class_counts.plot(kind="bar", ax=ax, color="steelblue")
ax.set_ylabel("Number of samples")
ax.set_xlabel("Medical Specialty")
ax.set_title("Class Distribution in Training Set")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("figures/class_distribution.png", dpi=150)
plt.show()
```

**Expected output you must report in the paper:**

| Specialty | Count | % |
| --- | --- | --- |
| Surgery | ~893 | 33.5% |
| Cardiovascular-Pulmonary | ~309 | 11.6% |
| Orthopedic | ~300 | 11.2% |
| Radiology | ~222 | 8.3% |
| General Medicine | ~212 | 7.9% |
| Gastroenterology | ~195 | 7.3% |
| Neurology | ~184 | 6.9% |
| Obstetrics-Gynecology | ~136 | 5.1% |
| Neurosurgery | ~79 | 3.0% |
| Ophthalmology | ~73 | 2.7% |
| Psychiatry-Psychology | ~41 | 1.5% |
| Dermatology | ~25 | 0.9% |

> **Concrete Observation #1 (write this in the paper):** “The dataset exhibits severe class imbalance: Surgery accounts for 33.5% of all samples while Dermatology comprises only 0.9%, a ratio of approximately 35:1. This imbalance poses a significant risk that classifiers will over-predict majority classes, potentially causing per-class F1 scores to drop below the 25% threshold required by the evaluation criteria.”
> 

## 1.2 Text Length Analysis

> **What to do:** Compute character/word lengths per field and per specialty.
**Why:** This reveals which fields carry the most text, whether transformer truncation will lose information, and whether text length correlates with specialty.
> 

```python
# Compute text length features
for col in ["description", "sample_name", "transcription", "keywords"]:
    train_df[f"{col}_len"] = train_df[col].apply(lambda x: len(str(x).split()))

# Summary statistics
print(train_df[["description_len", "sample_name_len", "transcription_len", "keywords_len"]].describe())

# ── Box plot of transcription length per specialty ──
fig, ax = plt.subplots(figsize=(12, 5))
train_df.boxplot(column="transcription_len", by="medical_specialty", ax=ax, rot=45)
ax.set_ylabel("Word count")
ax.set_title("Transcription Length by Specialty")
plt.tight_layout()
plt.savefig("figures/transcription_length_by_specialty.png", dpi=150)
plt.show()
```

> **Concrete Observation #2 (write this in the paper):** “Clinical transcriptions vary dramatically in length: Radiology reports average ~X words (short, structured), while Surgery transcriptions average ~Y words (long, detailed operative notes). This length disparity has direct implications for transformer-based models, which truncate inputs at 512 tokens — Surgery transcriptions may lose critical post-operative findings, while Radiology reports fit comfortably.”
> 

## 1.3 Keyword Leakage Analysis

> **What to do:** Check if the `keywords` field directly contains the specialty name.
**Why:** If keywords = “cardiovascular / pulmonary, …” for Cardiovascular-Pulmonary cases, then this field is essentially leaking the label. This is a critical observation to report and it directly impacts how you should design your experiments.
> 

```python
# Check how often keywords contain the specialty name (lowercased)
def keyword_contains_specialty(row):
    specialty_lower = row["medical_specialty"].lower().replace("-", " ")
    keywords_lower = str(row["keywords"]).lower().replace("-", " ")
    # Check if any significant word from specialty appears in keywords
    specialty_words = specialty_lower.split()
    return any(w in keywords_lower for w in specialty_words if len(w) > 3)

train_df["keyword_leakage"] = train_df.apply(keyword_contains_specialty, axis=1)
leakage_rate = train_df["keyword_leakage"].mean()
print(f"Keyword leakage rate:{leakage_rate:.1%}")

# Per specialty
print(train_df.groupby("medical_specialty")["keyword_leakage"].mean())
```

> **Concrete Observation #3 (bonus):** “We observed that the `keywords` field frequently contains tokens directly derived from the specialty name (e.g., ‘cardiovascular / pulmonary’ for Cardiovascular-Pulmonary cases). This occurs in X% of records, suggesting that models trained with keyword features may achieve artificially inflated accuracy by exploiting this shortcut rather than learning from clinical content. We therefore run ablation experiments with and without keywords to understand their true contribution.”
> 

## 1.4 Check for Missing Values and Annotation Noise

> **What to do:** Find records with missing fields and look for potentially mislabeled samples.
**Why:** The project description explicitly says the dataset has annotation discrepancies and will not be corrected. Finding and discussing these scores high on Data Analysis.
> 

```python
# Missing values
print("Missing values per column:")
print(train_df[["description", "sample_name", "transcription", "keywords"]].isnull().sum())
print("Empty string fields:")
for col in ["description", "sample_name", "transcription", "keywords"]:
    print(f"{col}:{(train_df[col] == '').sum()}")

# Look for potential mislabels: very short transcriptions
short_trans = train_df[train_df["transcription_len"] < 10]
print(f"\nRecords with < 10 words in transcription:{len(short_trans)}")
print(short_trans[["medical_specialty", "description", "transcription"]].head(10))
```

---

# Phase 2 — Data Preprocessing

> **Why this phase exists:** All models need clean, consistent input. Crucially, the project description warns: *“Attention to blindly removing stop words”* — in clinical text, words like “no”, “not”, “without”, “negative” carry diagnostic meaning. Removing them could flip the meaning of a sentence (e.g., “no evidence of fracture” → “evidence of fracture”).
> 

## 2.1 Text Combination Strategy

> **What to do:** Combine the 4 text fields into a single input string.
**Why:** Using only `description` (as the baseline does) throws away 80%+ of the available text. Each field carries different information: `description` is a summary, `sample_name` is a procedure title, `transcription` is the full note, `keywords` are tags. Combining them gives models maximum context.
> 

```python
def combine_text_fields(row, include_keywords=True):
    """Combine all text fields into a single string with field markers."""
    parts = [
        f"Title:{row['sample_name']}",
        f"Summary:{row['description']}",
        f"Clinical Note:{row['transcription']}",
    ]
    if include_keywords:
        parts.append(f"Keywords:{row['keywords']}")
    return " ".join(parts)

# Create combined text (with and without keywords for ablation)
train_df["text_all"] = train_df.apply(lambda r: combine_text_fields(r, include_keywords=True), axis=1)
train_df["text_no_kw"] = train_df.apply(lambda r: combine_text_fields(r, include_keywords=False), axis=1)

test_df["text_all"] = test_df.apply(lambda r: combine_text_fields(r, include_keywords=True), axis=1)
test_df["text_no_kw"] = test_df.apply(lambda r: combine_text_fields(r, include_keywords=False), axis=1)
```

## 2.2 Preprocessing for Classical Models

> **What to do:** Create a preprocessing pipeline that lowercases, stems, and removes noise — but **does NOT remove stop words** blindly.
**Why:** TF-IDF and Naive Bayes operate on raw token frequencies. Lowercasing and stemming reduce vocabulary size and improve generalization. But stop word removal in clinical text risks losing negation semantics.
> 

```python
import re
from nltk.stem import PorterStemmer

porter_stemmer = PorterStemmer()

def preprocess_for_classical(text):
    """Preprocess text for classical ML: lowercase, clean, stem (NO stop word removal)."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)      # Remove non-alpha characters
    text = re.sub(r"\s+", " ", text).strip()    # Collapse whitespace
    words = text.split()
    stemmed = [porter_stemmer.stem(w) for w in words]
    return " ".join(stemmed)

# Apply to training data
train_df["text_classical"] = train_df["text_all"].apply(preprocess_for_classical)

# IMPORTANT: Apply EXACT same preprocessing to test data
test_df["text_classical"] = test_df["text_all"].apply(preprocess_for_classical)
```

## 2.3 Train/Validation Split

> **What to do:** Create a stratified 80/20 train/validation split.
**Why:** Stratified split ensures each class has proportional representation in both sets. Without stratification, minority classes like Dermatology (25 samples) might end up with 0 samples in the validation set. The validation set is YOUR test set for all model evaluation — never touch `test_no_labels.csv` until generating `results.txt`.
> 

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train_df,
    train_df["label"],
    test_size=0.2,
    random_state=SEED,
    stratify=train_df["label"],  # stratified!
)

print(f"Train:{len(X_train)}, Val:{len(X_val)}")
print(f"Val label distribution:\n{y_val.value_counts().sort_index()}")
```

> ⚠️ **CRITICAL RULE (from template):** `test_no_labels.csv` must ONLY be used to generate `results.txt`. All model tuning, selection, and internal evaluation use YOUR validation split.
> 

---

# Phase 3 — Data Augmentation *(Section 2.2 — 2.0 pts)*

> **Why this phase exists:** Data augmentation is **mandatory** (the template explicitly says so). It is worth 2.0 points — the second-highest individual score item. You must clearly describe your strategy and its rationale.
> 
> 
> **Connection to RQ2:** A key sub-question is: *“Does data augmentation help classical models and transformers equally, or does one paradigm benefit more?”* This makes augmentation not just mandatory but central to your research argument.
> 

## 3.1 Strategy Selection

> **What to do:** Choose at least one augmentation technique. We recommend **synonym replacement** (simple, explainable, language-aware) and **random oversampling** (to directly address class imbalance) as a comparison pair.
> 

### Technique A: Synonym Replacement (via nlpaug)

> **Why:** Replaces words with synonyms from WordNet while preserving meaning. This generates diverse paraphrases of clinical texts. It is a content-aware augmentation that respects medical terminology.
> 

```python
import nlpaug.augmenter.word as naw

# Synonym replacement augmenter using WordNet
syn_aug = naw.SynonymAug(aug_src="wordnet", aug_min=1, aug_max=5, aug_p=0.1)

def augment_text_synonym(text, n_augments=1):
    """Generate n_augments augmented versions of the text."""
    augmented = []
    for _ in range(n_augments):
        aug_text = syn_aug.augment(text)
        if isinstance(aug_text, list):
            augmented.append(aug_text[0])
        else:
            augmented.append(aug_text)
    return augmented
```

### Technique B: Random Oversampling of Minority Classes

> **Why:** Directly duplicates samples from underrepresented classes. Simple, fast, and effective for ensuring the model sees enough examples of rare specialties. Combined with synonym replacement, the model sees both original AND paraphrased versions.
> 

```python
from sklearn.utils import resample

def oversample_minority_classes(df, target_col="label", target_count=None):
    """Oversample minority classes to match the majority class count (or target_count)."""
    class_counts = df[target_col].value_counts()
    if target_count is None:
        target_count = class_counts.max()

    frames = []
    for label, count in class_counts.items():
        class_df = df[df[target_col] == label]
        if count < target_count:
            oversampled = resample(class_df, replace=True, n_samples=target_count, random_state=SEED)
            frames.append(oversampled)
        else:
            frames.append(class_df)
    return pd.concat(frames, ignore_index=True)
```

## 3.2 Apply Augmentation to Training Set ONLY

> **What to do:** Augment ONLY the training split (not validation, not test).
**Why:** Augmenting validation or test data breaks the integrity of evaluation.
> 

```python
# Strategy: apply synonym replacement ONLY to minority classes (< median count)
median_count = X_train["label"].value_counts().median()
minority_classes = X_train["label"].value_counts()[X_train["label"].value_counts() < median_count].index.tolist()

augmented_rows = []
for _, row in X_train[X_train["label"].isin(minority_classes)].iterrows():
    aug_texts = augment_text_synonym(row["text_all"], n_augments=2)
    for aug_text in aug_texts:
        new_row = row.copy()
        new_row["text_all"] = aug_text
        new_row["text_classical"] = preprocess_for_classical(aug_text)
        augmented_rows.append(new_row)

X_train_augmented = pd.concat([X_train, pd.DataFrame(augmented_rows)], ignore_index=True)
print(f"Original train size:{len(X_train)}")
print(f"After augmentation:{len(X_train_augmented)}")
print(f"New class distribution:\n{X_train_augmented['label'].value_counts().sort_index()}")
```

## 3.3 What to Write in the Paper

In the report, clearly state:

1. **Strategy chosen:** synonym replacement + oversampling of minority classes
2. **Rationale:** the 35:1 class imbalance means models may never see enough examples of Dermatology, Psychiatry-Psychology, etc.; augmentation ensures the model sees diverse representations of minority classes
3. **Implementation:** used the `nlpaug` library with WordNet-based synonym replacement at a 10% word replacement rate; augmented minority classes (count < median) with 2 additional samples per record
4. **Impact on RQ2:** compare the effect of augmentation on classical models vs. transformers — does it help one paradigm more than the other? (Report these results in the Results section)

---

# Phase 4 — Model Development *(Section 3 — 4.0 pts)*

> **Why this phase exists:** The template requires **at least 3 models** (1.0 pt), described clearly enough to replicate (1.0 pt), with creative and thoughtful choices (2.0 pt). Since your RQ is “Classical ML vs. Transformers”, you naturally need models from both paradigms.
> 
> 
> **The models for RQ2:**
> 
> 1. **Model A — Multinomial NB + TF-IDF** (weak baseline, referenced in project description)
> 2. **Model B — LinearSVC + TF-IDF** (strong classical)
> 3. **Model C — Fine-tuned Bio_ClinicalBERT** (domain-specific transformer)
> 4. *(Optional Model D — Fine-tuned general bert-base-uncased for domain vs. general comparison)*

---

## 4.1 Model A: Multinomial Naive Bayes + TF-IDF (Weak Baseline)

> **Why this model:** The project description specifies this as the weak baseline to beat (accuracy ~49%). It must be included. It is also the simplest classical approach and establishes a lower bound.
> 
> 
> **How it works conceptually (write this in the paper):**
> 
> - **TF-IDF** converts each document into a fixed-size sparse vector where each dimension corresponds to a term in the vocabulary. Term Frequency (TF) measures how often a word appears in a document; Inverse Document Frequency (IDF) down-weights words that appear in many documents (e.g., “patient” appears in every clinical note and is thus not discriminative).
> - **Multinomial Naive Bayes** assumes features (TF-IDF values) are independent given the class and computes $P(\text{class} | \text{document})$ using Bayes’ theorem. Despite the naive independence assumption, it works surprisingly well for text classification due to the high dimensionality of text features.

### Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── Vectorize ──
# The baseline from the project description uses ONLY the 'description' field
# with lowercasing and stemming. We replicate this exactly.
tfidf_baseline = TfidfVectorizer(
    ngram_range=(1, 1),      # unigrams only (simplest baseline)
    max_features=10000,       # limit vocabulary
    min_df=2,                 # ignore very rare terms
    sublinear_tf=True,        # apply log normalization to TF
)

# Use only the preprocessed description for the baseline
X_train_tfidf_base = tfidf_baseline.fit_transform(
    X_train["description"].apply(preprocess_for_classical)
)
X_val_tfidf_base = tfidf_baseline.transform(
    X_val["description"].apply(preprocess_for_classical)
)

# ── Train ──
mnb = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing
mnb.fit(X_train_tfidf_base, y_train)

# ── Evaluate ──
y_pred_base = mnb.predict(X_val_tfidf_base)
print(f"Baseline Accuracy:{accuracy_score(y_val, y_pred_base):.4f}")
print(classification_report(y_val, y_pred_base, target_names=label_encoder.classes_))
```

> **Hyperparameters to document:**
> 
> - Vectorizer: TF-IDF, unigrams, max_features=10000, min_df=2, sublinear_tf=True
> - Classifier: MultinomialNB, alpha=1.0 (Laplace smoothing)
> - Input: `description` field only, lowercased + stemmed

---

## 4.2 Model B: LinearSVC + TF-IDF (Strong Classical)

> **Why this model:** SVM with a linear kernel is consistently one of the strongest classical methods for text classification. Using TF-IDF with both word and character n-grams captures both semantic and morphological patterns (e.g., “cholecystectomy”, “gastroenterology” share character n-grams with their respective specialties). This represents the best the classical ML paradigm can offer.
> 
> 
> **How it works conceptually (write this in the paper):**
> 
> - **LinearSVC** finds the hyperplane in TF-IDF feature space that maximizes the margin between classes. For multi-class, it uses a one-vs-rest strategy (12 binary classifiers).
> - **Character n-grams (3–5)** capture sub-word patterns like “ectom” (related to surgical removal), “scop” (related to endoscopy), “cardio” (cardiovascular). These patterns are highly discriminative for medical specialties.
> - This model uses ALL text fields combined, not just `description`.

### Implementation

```python
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# ── Strong TF-IDF: word + character n-grams on ALL fields ──
tfidf_strong = TfidfVectorizer(
    ngram_range=(1, 2),          # word unigrams and bigrams
    max_features=50000,
    min_df=2,
    sublinear_tf=True,
    analyzer="word",
)

tfidf_char = TfidfVectorizer(
    ngram_range=(3, 5),          # character 3-to-5-grams
    max_features=50000,
    min_df=2,
    sublinear_tf=True,
    analyzer="char_wb",          # only at word boundaries
)

# Combine word and char features
from sklearn.pipeline import FeatureUnion

combined_features = FeatureUnion([
    ("word_tfidf", tfidf_strong),
    ("char_tfidf", tfidf_char),
])

# ── Use all combined text (preprocessed) ──
X_train_combined = combined_features.fit_transform(X_train["text_classical"])
X_val_combined = combined_features.transform(X_val["text_classical"])

# ── Train LinearSVC ──
svc = LinearSVC(
    C=1.0,              # regularization parameter
    max_iter=10000,     # ensure convergence
    random_state=SEED,
    class_weight="balanced",  # automatically adjust weights inversely proportional to class frequency
)
svc.fit(X_train_combined, y_train)

# ── Evaluate ──
y_pred_svc = svc.predict(X_val_combined)
print(f"LinearSVC Accuracy:{accuracy_score(y_val, y_pred_svc):.4f}")
print(classification_report(y_val, y_pred_svc, target_names=label_encoder.classes_))
```

> **Hyperparameters to document:**
> 
> - Word TF-IDF: (1,2)-grams, max_features=50000, min_df=2, sublinear_tf
> - Char TF-IDF: (3,5)-grams, char_wb analyzer, max_features=50000
> - Classifier: LinearSVC, C=1.0, class_weight=“balanced”
> - Input: all 4 fields combined, lowercased + stemmed

### 4.2.1 Ablation: With vs. Without Augmented Data

> **Why:** This directly answers the RQ sub-question: “Does augmentation help classical models?”
> 

```python
# Retrain on augmented data
X_train_aug_combined = combined_features.fit_transform(X_train_augmented["text_classical"])
X_val_aug_combined = combined_features.transform(X_val["text_classical"])

svc_aug = LinearSVC(C=1.0, max_iter=10000, random_state=SEED, class_weight="balanced")
svc_aug.fit(X_train_aug_combined, X_train_augmented["label"])

y_pred_svc_aug = svc_aug.predict(X_val_aug_combined)
print(f"LinearSVC (augmented) Accuracy:{accuracy_score(y_val, y_pred_svc_aug):.4f}")
print(classification_report(y_val, y_pred_svc_aug, target_names=label_encoder.classes_))
```

---

## 4.3 Model C: Fine-tuned Bio_ClinicalBERT (Domain-Specific Transformer)

> **Why this model:** This is the other side of the RQ — a state-of-the-art transformer pre-trained specifically on clinical text (MIMIC-III clinical notes). It represents the best the transformer paradigm can offer for this domain. Bio_ClinicalBERT has already learned medical vocabulary, abbreviations, and clinical document structure during pre-training.
> 
> 
> **How it works conceptually (write this in the paper):**
> 
> - **Pre-training:** Bio_ClinicalBERT was initialized from BioBERT (trained on PubMed abstracts) and then further pre-trained on ~2 million clinical notes from MIMIC-III. This gives it deep understanding of clinical language.
> - **Fine-tuning:** We add a classification head (a single linear layer) on top of the `[CLS]` token representation and train the entire model end-to-end on our labeled clinical notes. The model’s pre-existing knowledge of clinical language accelerates learning and improves generalization.
> - **Tokenization:** BERT uses WordPiece subword tokenization (splits unknown words into known sub-pieces), with `[CLS]` at the start and `[SEP]` at the end. Maximum sequence length is 512 tokens — longer documents are truncated.

### Implementation

> **Note:** This follows the EXACT patterns from Lab P8 (`P8_Notebook_finetuning.ipynb`), adapted for our dataset.
> 

```python
import torch
import transformers
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
import evaluate
import numpy as np

# ── Choose the model ──
# For the main experiment, use Bio_ClinicalBERT (domain-specific)
model_name = "emilyalsentzer/Bio_ClinicalBERT"
# For optional comparison with general model, also run with:
# model_name = "bert-base-uncased"

# ── Tokenizer ──
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# ── Prepare HuggingFace Datasets ──
# Need: "text" and "label" columns
hf_train = Dataset.from_pandas(
    X_train[["text_all", "label"]].rename(columns={"text_all": "text"}).reset_index(drop=True)
)
hf_val = Dataset.from_pandas(
    X_val[["text_all", "label"]].rename(columns={"text_all": "text"}).reset_index(drop=True)
)

# ── Tokenize ──
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,        # truncate to max_length (512)
        max_length=512,
        padding=False,          # dynamic padding via DataCollator (more efficient)
    )

tokenized_train = hf_train.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_val = hf_val.map(tokenize_fn, batched=True, remove_columns=["text"])

# ── Model ──
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=n_classes,       # 12 medical specialties
)

# ── Metrics ──
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
        "f1_weighted": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

# ── Data Collator (dynamic padding) ──
data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

# ── Training Arguments ──
training_args = transformers.TrainingArguments(
    output_dir="results/bio_clinical_bert",
    learning_rate=2e-5,                   # standard for BERT fine-tuning
    per_device_train_batch_size=8,        # may need to reduce to 4 if OOM
    per_device_eval_batch_size=8,
    num_train_epochs=5,                   # 3-5 epochs is typical for fine-tuning
    weight_decay=0.01,                    # L2 regularization
    eval_strategy="epoch",               # evaluate after each epoch
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,          # restore best checkpoint
    metric_for_best_model="f1_macro",     # select best model by macro-F1
    seed=SEED,
    data_seed=SEED,
    report_to="none",                     # disable wandb/mlflow
)

# ── Trainer ──
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ── Train! ──
trainer.train()

# ── Evaluate on validation set ──
eval_results = trainer.evaluate()
print("Eval results:", eval_results)

# ── Detailed predictions ──
predictions = trainer.predict(tokenized_val)
y_pred_bert = np.argmax(predictions.predictions, axis=-1)
y_true_bert = predictions.label_ids

print(f"Bio_ClinicalBERT Accuracy:{accuracy_score(y_true_bert, y_pred_bert):.4f}")
print(classification_report(y_true_bert, y_pred_bert, target_names=label_encoder.classes_))
```

> **Hyperparameters to document (ALL of these — for replicability):**
> 
> - Model: `emilyalsentzer/Bio_ClinicalBERT`
> - Tokenizer: WordPiece, max_length=512, truncation=True
> - Learning rate: 2e-5
> - Batch size: 8
> - Epochs: 5
> - Weight decay: 0.01
> - Optimizer: AdamW (default in Trainer)
> - Best model selection: by macro-F1 on validation
> - GPU: specify what GPU you used (e.g., “NVIDIA T4 on Google Colab”)

---

## 4.4 *(Optional)* Model D: Fine-tuned bert-base-uncased (General Transformer)

> **Why (for creativity):** Running the same fine-tuning setup with `bert-base-uncased` instead of `Bio_ClinicalBERT` lets you directly answer: “Does domain-specific pre-training matter?” This adds depth to the Classical vs. Transformer comparison and earns creativity points.
> 

```python
# Repeat the EXACT same pipeline as Model C but with:
model_name_general = "bert-base-uncased"
# ... (same code, different model_name)
```

---

# Phase 5 — Experimental Setup *(Section 4 — 1.0 pt)*

> **Why this phase exists:** The template says: *“Provide clear information regarding the used datasets, splits, evaluation metrics used and hyperparameters.”*
This section is short in the paper (~0.25 pages) but must be precise.
> 

## 5.1 What to Write

In the paper, document:

### Datasets

- Training set: 2,669 samples across 12 medical specialties (semicolon-delimited CSV with 5 fields)
- Own validation set: 20% stratified split (seed=42) → ~534 samples for internal evaluation
- Held-out test: 397 samples (`test_no_labels.csv`) — used ONLY for `results.txt` generation

### Evaluation Metrics

> **Why these metrics:** Accuracy alone is misleading with class imbalance (a model predicting “Surgery” for everything achieves 33.5% accuracy). Macro-F1 treats all classes equally regardless of size. Weighted-F1 weights by class frequency. Per-class F1 reveals which specialties are problematic.
> 
- **Accuracy** — overall correct predictions / total predictions
- **Macro-F1** — unweighted mean of per-class F1 scores (treats all 12 classes equally)
- **Weighted-F1** — weighted mean of per-class F1 by support
- **Per-class Precision, Recall, F1** — for the best model
- **Confusion Matrix** — for at least one model

### Hyperparameters

- List ALL hyperparameters for EACH model (see the code sections above)
- Briefly explain what each hyperparameter controls (e.g., “C controls the regularization strength in SVM — higher C penalizes misclassifications more”)

---

# Phase 6 — Results *(Section 5 — 1.5 pts)*

> **Why this phase exists:** The template requires: (1) results of best models, (2) per-label results of the best model, (3) a comprehensive confusion matrix. The test set must ONLY be used for `results.txt`.
> 

## 6.1 Comparison Table

> **What to do:** Create a clear table comparing all models on the same metrics.
> 

```python
# Collect all results into a summary table
results = {
    "Model": [
        "MNB + TF-IDF (description only)",
        "LinearSVC + TF-IDF (all fields)",
        "LinearSVC + TF-IDF (augmented)",
        "Bio_ClinicalBERT (all fields)",
    ],
    "Accuracy": [
        accuracy_score(y_val, y_pred_base),
        accuracy_score(y_val, y_pred_svc),
        accuracy_score(y_val, y_pred_svc_aug),
        accuracy_score(y_true_bert, y_pred_bert),
    ],
    # ... add Macro-F1, Weighted-F1
}
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```

**Table format for the paper:**

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- |
| MNB + TF-IDF (description) | X% | X% | X% |
| LinearSVC + TF-IDF (all fields) | X% | X% | X% |
| LinearSVC + TF-IDF (augmented) | X% | X% | X% |
| Bio_ClinicalBERT (all fields) | X% | X% | X% |

## 6.2 Per-Class Results of Best Model

```python
# Assuming best model is Bio_ClinicalBERT:
report = classification_report(
    y_true_bert, y_pred_bert,
    target_names=label_encoder.classes_,
    output_dict=True,
)
per_class_df = pd.DataFrame(report).transpose()
print(per_class_df)
```

**Table format for the paper (for best model):**

| Specialty | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Surgery | … | … | … | … |
| … | … | … | … | … |
| Dermatology | … | … | … | … |

## 6.3 Confusion Matrix

> **What to do:** Generate a confusion matrix for at least one model. Use the abbreviated specialty names so they are readable.
> 

```python
from sklearn.metrics import ConfusionMatrixDisplay

# Short labels for readability
short_labels = [name[:8] for name in label_encoder.classes_]

fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_true_bert, y_pred_bert)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_labels)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
ax.set_title("Confusion Matrix — Bio_ClinicalBERT")
plt.tight_layout()
plt.savefig("figures/confusion_matrix_bert.png", dpi=150)
plt.show()
```

> **Template tip:** “Use labels that are easy to be read in the Confusion Matrix.” → Abbreviate long specialty names.
> 

## 6.4 Effect of Augmentation Comparison

> **Why:** This directly addresses part of RQ2 — did augmentation benefit classical models differently than it would benefit transformers?
> 

**Table format:**

| Model | Without Augmentation (Macro-F1) | With Augmentation (Macro-F1) | Δ |
| --- | --- | --- | --- |
| LinearSVC + TF-IDF | X% | Y% | +Z% |
| Bio_ClinicalBERT | X% | Y% | +Z% |

## 6.5 Verify Automatic Evaluation Hurdles

```python
# ── Hurdle 1: Accuracy > 49% ──
print(f"Best accuracy:{max_accuracy:.4f} (need > 0.49)")

# ── Hurdle 2: No per-class F1 < 25% ──
for cls, metrics in report.items():
    if cls in label_encoder.classes_:
        if metrics["f1-score"] < 0.25:
            print(f"WARNING:{cls} F1 ={metrics['f1-score']:.3f} < 0.25!")
```

---

# Phase 7 — Discussion & Error Analysis *(Section 6 — 3.0 pts)*

> **Why this phase exists:** This is the HIGHEST-SCORING section (3.0 pts). The template explicitly says:
> 
> - *“Show us that you have properly analysed the dataset and the obtained outputs”*
> - *“Present and discuss at least 3 concrete examples of misclassified instances”*
> - *“Generic statements score zero.”*
> - *“Explain how results align with the research question”*
> 
> This is where your paper lives or dies. You must show you looked at actual data, not just metrics.
> 

## 7.1 Confusion Pattern Analysis

> **What to do:** Identify the 3-5 most common confusion pairs from the confusion matrix.
> 

```python
# Find the most confused pairs (off-diagonal elements)
import numpy as np

# Zero out diagonal
cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)

# Find top 5 confusion pairs
top_confusions = []
for _ in range(5):
    idx = np.unravel_index(cm_off.argmax(), cm_off.shape)
    top_confusions.append({
        "true": label_encoder.classes_[idx[0]],
        "predicted": label_encoder.classes_[idx[1]],
        "count": cm_off[idx],
    })
    cm_off[idx] = 0

for conf in top_confusions:
    print(f"{conf['true']} → predicted as{conf['predicted']}:{conf['count']} times")
```

> **Example analysis to write:** “The most common confusion pair is Neurology ↔︎ Neurosurgery (X instances). This is clinically expected: both specialties deal with the nervous system, and a brain tumor case may be classified under Neurology (diagnosis) or Neurosurgery (surgical treatment). The shared vocabulary — ‘cranial’, ‘cerebral’, ‘neurological’ — makes these cases inherently ambiguous even for human annotators.”
> 

## 7.2 Three Concrete Misclassified Examples

> **What to do:** Find 3 actual misclassified records, quote the text, and explain WHY the model got confused.
> 

```python
# Get misclassified samples from validation set
misclassified_mask = y_pred_bert != y_true_bert
misclassified_df = X_val[misclassified_mask].copy()
misclassified_df["predicted"] = label_encoder.inverse_transform(y_pred_bert[misclassified_mask])
misclassified_df["true_label"] = label_encoder.inverse_transform(y_true_bert[misclassified_mask])

# Display 5 examples to choose from
for i, (_, row) in enumerate(misclassified_df.head(5).iterrows()):
    print(f"\n{'='*80}")
    print(f"Example{i+1}")
    print(f"  TRUE:{row['true_label']}")
    print(f"  PREDICTED:{row['predicted']}")
    print(f"  Description:{row['description'][:200]}...")
    print(f"  Transcription:{row['transcription'][:300]}...")
```

**For the paper, format EACH example like this:**

> **Example 1:**
> 
> - **Text excerpt:** “PREOPERATIVE DIAGNOSIS: Right inguinal hernia. POSTOPERATIVE DIAGNOSIS: Right inguinal hernia. PROCEDURE PERFORMED: Right inguinal hernia repair with mesh…”
> - **True label:** Surgery
> - **Predicted label:** Gastroenterology
> - **Interpretation:** “This surgical case was misclassified as Gastroenterology likely because inguinal hernia repair involves the abdominal region, and terms like ‘abdomen’, ‘inguinal’, and ‘incision’ frequently appear in gastroenterology reports. The classical SVM model, which relies on TF-IDF features, confuses these shared terms. Notably, Bio_ClinicalBERT correctly classified this case, suggesting that the transformer’s contextual understanding of ‘hernia repair with mesh’ as a surgical procedure provides an advantage over bag-of-words representations.”

## 7.3 Tying Results Back to the Research Question

> **What to do:** Explicitly answer RQ2 using your experimental evidence.
> 

**Structure for this paragraph:**

1. **Overall comparison:** “Bio_ClinicalBERT achieved X% macro-F1 vs. LinearSVC’s Y% — a Z percentage point improvement, demonstrating that transformers outperform classical ML on this clinical text classification task.”
2. **Where classical excels (if anywhere):** “However, LinearSVC achieved higher per-class F1 for [Specialty X], suggesting that when discriminative n-gram patterns are clear (e.g., ‘fracture’, ‘orthopedic’ in Orthopedic cases), handcrafted features can be sufficient.”
3. **Where transformers excel:** “Bio_ClinicalBERT showed the largest advantage over LinearSVC for [Neurology vs. Neurosurgery], where contextual understanding of clinical narrative structure (e.g., ‘patient referred for surgical consultation’ vs. ‘neurological examination normal’) is critical.”
4. **Augmentation interaction:** “Data augmentation improved LinearSVC macro-F1 by X% but Bio_ClinicalBERT by only Y%, suggesting that pre-trained transformers are inherently more robust to class imbalance due to their rich pre-trained representations.”
5. **Acknowledge limitations:** “These findings are based on a dataset of ~2,669 records from MTSamples. The relatively small size may advantage classical models (which are less prone to overfitting on small datasets), potentially understating the transformer advantage that would emerge with larger clinical corpora.”

---

# Phase 8 — Future Work *(Section 7 — 0.5 pts)*

> **What to write:** Frame future work around the RQ. Examples:
> 
> - “With more time, we would explore **ensemble methods** combining the complementary strengths of classical and transformer models — using SVM predictions for specialties where n-grams are sufficient and BERT predictions for ambiguous cases.”
> - “We would investigate **long-context models** (e.g., Longformer, Clinical-Longformer) to handle Surgery transcriptions that exceed BERT’s 512-token limit.”
> - “A **hierarchical classification** approach (first classifying into surgical vs. non-surgical, then sub-classifying) could reduce confusion between closely related specialties.”
> - “More sophisticated **data augmentation** strategies such as back-translation or LLM-based paraphrasing could further improve minority class performance.”

---

# Phase 9 — Write the Paper

> **Paper Constraints:**
> 
> - **Max 3 pages** (only the first 3 are read/evaluated!)
> - Bibliography does NOT count toward 3 pages
> - Appendix: max 1 page, does NOT count toward 3 pages, paper must be self-contained without it
> - Language: English or Portuguese
> - Template: `template.tex` (ACL-style, twocolumn, 10pt)
> - No contractions, no subjective adjectives, formal scientific English

## 9.1 Section Length Budget (3 pages, two-column)

| Section | Suggested Budget |
| --- | --- |
| Introduction | ~0.4 pages |
| Data (2.1 + 2.2) | ~0.5 pages |
| Models | ~0.7 pages |
| Experimental Setup | ~0.2 pages |
| Results | ~0.5 pages |
| Discussion | ~0.5 pages |
| Future Work | ~0.1 pages |

## 9.2 Checklist

- [ ]  Title is informative (e.g., “Classical vs. Transformer-Based Medical Specialty Classification”)
- [ ]  Authors include contribution percentages (mandatory!)
- [ ]  Introduction clearly states the RQ
- [ ]  Data Analysis has ≥2 concrete, non-generic observations with numbers
- [ ]  Data Augmentation strategy is described with rationale
- [ ]  ≥3 models are described clearly with all hyperparameters
- [ ]  Experimental setup documents splits, metrics, seeds
- [ ]  Results include comparison table, per-class table, confusion matrix
- [ ]  Discussion has ≥3 concrete misclassified examples with text excerpts
- [ ]  Discussion ties back to the RQ explicitly
- [ ]  Future work is framed around the RQ
- [ ]  All figures/tables are numbered, captioned, and referenced in text
- [ ]  No contractions (it’s → it is)
- [ ]  No subjective adjectives (amazing, nice, great)
- [ ]  LaTeX quotation marks: ``like this’’
- [ ]  All sources cited: papers, libraries, pre-trained models
- [ ]  LLM usage explicitly declared
- [ ]  Paper is ≤3 pages (excluding bibliography and appendix)
- [ ]  Confusion matrix labels are readable

---

# Phase 10 — Generate Predictions & Submit

## 10.1 Generate `results.txt`

> **CRITICAL:** Apply the EXACT same preprocessing to test data. The number of lines in `results.txt` must equal the number of test samples (397).
> 

```python
# ── For best model (e.g., Bio_ClinicalBERT) ──

# 1. Prepare test data
test_df["text_all"] = test_df.apply(
    lambda r: combine_text_fields(r, include_keywords=True), axis=1
)

# 2. Tokenize
hf_test = Dataset.from_pandas(
    test_df[["text_all"]].rename(columns={"text_all": "text"}).reset_index(drop=True)
)
tokenized_test = hf_test.map(tokenize_fn, batched=True, remove_columns=["text"])

# 3. Predict
test_predictions = trainer.predict(tokenized_test)
y_test_pred = np.argmax(test_predictions.predictions, axis=-1)

# 4. Convert back to label names
test_labels = label_encoder.inverse_transform(y_test_pred)

# 5. Write results.txt (NO HEADER, one label per line)
with open("results.txt", "w") as f:
    for label in test_labels:
        f.write(f"{label}\n")

# 6. Verify line count
with open("results.txt") as f:
    lines = f.readlines()
print(f"results.txt has{len(lines)} lines (expect{len(test_df)})")
assert len(lines) == len(test_df), "LINE COUNT MISMATCH!"
```

## 10.2 Package Submission

```
NUM.zip
├── NUM.pdf                 # The paper (max 3 pages + bibliography + appendix)
├── results.txt             # Predictions (one label per line, no header, 397 lines)
├── src/                    # Source code
│   ├── data_analysis.py    # Phase 1 code
│   ├── preprocessing.py    # Phase 2 code
│   ├── augmentation.py     # Phase 3 code
│   ├── model_baseline.py   # Model A (MNB)
│   ├── model_svm.py        # Model B (SVC)
│   ├── model_bert.py       # Model C (Bio_ClinicalBERT)
│   ├── evaluate.py         # Results generation
│   ├── generate_results.py # results.txt generation
│   └── requirements.txt    # Library versions
└── (any external datasets collected)
```

## 10.3 Final Verification

- [ ]  `results.txt` has exactly 397 lines
- [ ]  `results.txt` has NO header
- [ ]  Each line is a valid specialty name (one of the 12)
- [ ]  Paper is ≤ 3 pages
- [ ]  File is `.zip` NOT `.rar`
- [ ]  Named `NUM.zip` (group number)
- [ ]  Code runs end-to-end and reproduces paper numbers
- [ ]  Contribution percentages are in the author line

---

> [!CAUTION]
**Critical Pitfalls:**
> 
> 1. `test_no_labels.csv` has **NO HEADER** — use `header=None, names=[...]`
> 2. **Never** use test data during training/validation/tuning
> 3. **Do NOT blindly remove stop words** — negation words carry clinical meaning
> 4. Paper > 3 pages → only first 3 read → you lose content
> 5. Missing LLM declaration → penalty on References
> 6. Generic statements in Discussion → **scores zero**
> 7. Generic data analysis → **scores zero**
> 8. ANY instruction violation → **3.0 points**
> 9. Cannot replicate results in oral defense → **0 points** for automatic evaluation