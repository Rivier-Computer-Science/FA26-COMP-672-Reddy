import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Configuration & Reproducibility
SEED = 42
TARGET_NAMES = ["Joy", "Anger", "Sadness", "Fear", "Neutral"]
OUTPUT_DIR = PROJECT_ROOT/ "outputs" / "baseline"
C_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]  # Hyperparameter search space on validation set


def validate_text_input(text: str) -> str:
    """Handles boundary cases (None, non-string, empty, or whitespace-only text)."""
    if not isinstance(text, str) or not text.strip():
        return "[EMPTY_INPUT]"
    return text.strip()


def load_split(split_name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"data_{split_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run src/prepare_data.py first to generate deterministic splits."
        )
    df = pd.read_csv(path)
    df["clean_text"] = df["text"].apply(validate_text_input)
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load fixed deterministic splits
    train_df = load_split("train")
    val_df = load_split("validation")
    test_df = load_split("test")

    # Fit TF-IDF strictly on training text (Chapters 4 & 6)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # Unigrams + bigrams capture basic local negation (e.g., "not happy")
        max_features=25000,      # Cap vocabulary size for memory/speed efficiency
        sublinear_tf=True,       # Apply 1 + log(tf) scaling
        strip_accents="unicode",
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_val = vectorizer.transform(val_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    # Tune regularization strength C strictly on validation Macro F1
    best_c = None
    best_val_f1 = -1.0
    best_model = None
    tuning_logs = []

    print("=== Tuning Logistic Regression C on Validation Split ===")
    for c_val in C_GRID:
        clf = LogisticRegression(
            C=c_val,
            class_weight="balanced",  # Mitigates class imbalance using inverse class frequencies
            max_iter=1000,
            random_state=SEED,
        )
        clf.fit(X_train, y_train)
        val_preds = clf.predict(X_val)
        val_macro_f1 = f1_score(y_val, val_preds, average="macro")
        val_acc = accuracy_score(y_val, val_preds)

        tuning_logs.append({
            "C": c_val,
            "val_macro_f1": round(float(val_macro_f1), 4),
            "val_accuracy": round(float(val_acc), 4),
        })
        print(f"C={c_val:<4} | Val Macro F1: {val_macro_f1:.4f} | Val Acc: {val_acc:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_c = c_val
            best_model = clf

    print(f"\nSelected Best Hyperparameter: C={best_c} (Val Macro F1: {best_val_f1:.4f})")

    # Locked Final Evaluation on Held-Out Test Split (with Latency Timing)
    start_time = time.perf_counter()
    test_probs = best_model.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    latency_per_100 = (elapsed_ms / len(test_df)) * 100.0

    # Check how many test examples had zero known vocabulary tokens (OOV / emoji-only boundary)
    zero_token_mask = np.asarray(X_test.sum(axis=1)).ravel() == 0
    oov_count = int(zero_token_mask.sum())

    test_macro_f1 = f1_score(y_test, test_preds, average="macro")
    test_acc = accuracy_score(y_test, test_preds)
    report_dict = classification_report(
        y_test, test_preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, test_preds).tolist()

    print("\n=== Locked Test Set Results (TF-IDF + Logistic Regression) ===")
    print(classification_report(y_test, test_preds, target_names=TARGET_NAMES, digits=4, zero_division=0))
    print(f"Test Macro F1: {test_macro_f1:.4f} | Test Accuracy: {test_acc:.4f}")
    print(f"Inference Latency: {latency_per_100:.3f} ms per 100 comments")
    print(f"Zero-Token (OOV/Emoji-only) Test Examples: {oov_count} / {len(test_df)}")

    # Save Predictions with Example IDs (Required for Rubric Verification & Bootstrap)
    pred_df = pd.DataFrame({
        "id": test_df["id"],
        "text": test_df["text"],
        "true_label": y_test,
        "true_label_name": [TARGET_NAMES[i] for i in y_test],
        "pred_label": test_preds,
        "pred_label_name": [TARGET_NAMES[i] for i in test_preds],
        "is_oov_boundary": zero_token_mask,
    })
    for idx, name in enumerate(TARGET_NAMES):
        pred_df[f"prob_{name.lower()}"] = np.round(test_probs[:, idx], 6)

    pred_path = os.path.join(OUTPUT_DIR, "predictions_baseline_test.csv")
    pred_df.to_csv(pred_path, index=False)

    # Save Reproducible Metrics & Config JSON
    metrics_payload = {
        "model": "TF-IDF (1,2)-gram + Balanced Logistic Regression",
        "seed": SEED,
        "best_C": best_c,
        "vocab_size": len(vectorizer.vocabulary_),
        "validation_tuning_logs": tuning_logs,
        "test_macro_f1": round(float(test_macro_f1), 4),
        "test_accuracy": round(float(test_acc), 4),
        "latency_ms_per_100_examples": round(float(latency_per_100), 4),
        "oov_zero_token_test_examples": oov_count,
        "per_class_report": report_dict,
        "confusion_matrix": cm,
    }
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_baseline.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Boundary Case Demonstration (Required by Rubric Section 3)
    print("\n=== Boundary Case Sanity Checks ===")
    boundary_samples = ["", "   ", "😂😂😂", "xyzqwk unknownword"]
    for sample in boundary_samples:
        cleaned = validate_text_input(sample)
        vec = vectorizer.transform([cleaned])
        probs = best_model.predict_proba(vec)[0]
        pred_idx = int(np.argmax(probs))
        print(
            f"Input: {repr(sample):<22} -> Active TF-IDF Features: {vec.nnz:<2} | "
            f"Pred: {TARGET_NAMES[pred_idx]} ({probs[pred_idx]:.2f})"
        )


if __name__ == "__main__":
    main()