import os
import argparse
import torch
import numpy as np
import pandas as pd
import transformers

from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
import evaluate

from utils import SEED, set_seed, load_data, encode_labels, create_train_val_split
from preprocessing import apply_text_combination


def prepare_hf_dataset(df: pd.DataFrame, text_col: str = "text_all", label_col: str = "label"):
    """Converts pandas DataFrame into a HuggingFace Dataset with 'text' and 'label'."""

    subset = df[[text_col, label_col]].rename(columns={text_col: "text"}).reset_index(drop=True)
    return Dataset.from_pandas(subset)


def train_evaluate_transformer(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    label_encoder,
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    output_dir: str = "results/bio_clinical_bert",
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    text_col: str = "text_all",
):
    """
    Fine-tunes a transformer model (e.g. Bio_ClinicalBERT) on medical specialty classification.
    Compatible with both Model C(Bio-ClinicalBERT) and Model D (bert-base-uncased).
    """
    
    n_classes = len(label_encoder.classes_)

    # 1. Load Tokenizer
    print(f"Loading tokenizer for '{model_name}'...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # 2. Prepare HuggingFace Datasets
    hf_train = prepare_hf_dataset(X_train, text_col=text_col)
    hf_val = prepare_hf_dataset(X_val, text_col=text_col)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding=False,  # dynamic padding via DataCollator
        )

    print("Tokenizing datasets (max_length=512)...")
    tokenized_train = hf_train.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_val = hf_val.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 3. Model Architecture
    print(f"Loading model architecture: {model_name} with {n_classes} target labels...")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=n_classes,
    )

    # 4. Metrics
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

    # 5. Data Collator (dynamic padding to longest sequence in each batch)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    # 6. Training Arguments
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=use_fp16,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )

    # 7. Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 8. Train & Evaluate
    print(f"Starting fine-tuning for {model_name} (fp16={use_fp16})...")
    trainer.train()

    # Predictions on validation set
    predictions = trainer.predict(tokenized_val)
    y_pred_bert = np.argmax(predictions.predictions, axis=-1)
    y_true_bert = predictions.label_ids

    acc = accuracy_score(y_true_bert, y_pred_bert)

    acc = accuracy_score(y_true_bert, y_pred_bert)
    macro_f1 = f1_metric.compute(predictions=y_pred_bert, references=y_true_bert, average="macro")["f1"]
    weighted_f1 = f1_metric.compute(predictions=y_pred_bert, references=y_true_bert, average="weighted")["f1"]

    
    print("\n" + "=" * 60)
    print(f" Evaluation: {model_name}")
    print("=" * 60)
    print(f"Validation Accuracy:    {acc:.4f} (~{acc*100:.2f}%)")
    print(f"Validation Macro F1:    {macro_f1:.4f}")
    print(f"Validation Weighted F1: {weighted_f1:.4f}\n")
    print("Detailed Classification Report:")
    print(classification_report(y_true_bert, y_pred_bert, target_names=label_encoder.classes_, zero_division=0))

    return {
        "trainer": trainer,
        "model": model,
        "tokenizer": tokenizer,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "y_pred": y_pred_bert,
        "y_true": y_true_bert,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fine-tune Transformers on Clinical Specialties")
    parser.add_argument(
        "--model",
        type=str,
        default="emilyalsentzer/Bio_ClinicalBERT",
        choices=["emilyalsentzer/Bio_ClinicalBERT", "bert-base-uncased", "all"],
        help="Transformer model to fine-tune (or 'all' for both C and D)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    args = parser.parse_args()

    set_seed(SEED)

    # 1. Load data & clean labels
    train_df, test_df = load_data()
    train_df, label_encoder = encode_labels(train_df)
    train_df = apply_text_combination(train_df)

    # 2. Stratified 80/20 train/validation split
    X_train, X_val, y_train, y_val = create_train_val_split(train_df)


    # 3. Models to run
    models_to_run = []
    if args.model == "all":
        models_to_run = [
            ("emilyalsentzer/Bio_ClinicalBERT", "results/bio_clinical_bert"),
            ("bert-base-uncased", "results/bert_base_uncased"),
        ]
    elif args.model == "bert-base-uncased":
        models_to_run = [("bert-base-uncased", "results/bert_base_uncased")]
    else:
        models_to_run = [("emilyalsentzer/Bio_ClinicalBERT", "results/bio_clinical_bert")]

    for m_name, out_dir in models_to_run:
            train_evaluate_transformer(
                X_train=X_train,
                X_val=X_val,
                label_encoder=label_encoder,
                model_name=m_name,
                output_dir=out_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
            )