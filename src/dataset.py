import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = "main"  # Recorded for rubric reproducibility


def validate_text_input(text: str) -> tuple[str, bool]:
    """
    Handles boundary cases (None, non-string, empty, or whitespace-only text).
    Returns (cleaned_text, is_empty_boundary).
    """
    if not isinstance(text, str) or not text.strip():
        return "[EMPTY]", True
    return text.strip(), False


def get_tokenizer(model_name: str = MODEL_NAME, revision: str = MODEL_REVISION):
    """Loads the pretrained WordPiece tokenizer unchanged."""
    return AutoTokenizer.from_pretrained(model_name, revision=revision)


class GoEmotionsDataset(Dataset):
    """PyTorch Dataset wrapping deterministic 5-class GoEmotions CSV splits."""

    def __init__(self, csv_path: str, tokenizer, max_length: int = 64):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Missing {csv_path}. Run src/prepare_data.py first."
            )
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-validate boundary cases
        cleaned_results = [validate_text_input(t) for t in self.df["text"]]
        self.texts = [res[0] for res in cleaned_results]
        self.empty_flags = [res[1] for res in cleaned_results]
        self.labels = self.df["label"].tolist()
        self.ids = self.df["id"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        # Check untruncated token length for failure-analysis diagnostics
        raw_tokens = self.tokenizer.encode(text, truncation=False)
        was_truncated = len(raw_tokens) > self.max_length

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "id": self.ids[idx],
            "input_ids": encoded["input_ids"].squeeze(0),           # Shape: (max_length,)
            "attention_mask": encoded["attention_mask"].squeeze(0), # Shape: (max_length,)
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "is_empty_boundary": torch.tensor(self.empty_flags[idx], dtype=torch.bool),
            "was_truncated": torch.tensor(was_truncated, dtype=torch.bool),
        }


def compute_class_weights(train_csv_path: str, num_classes: int = 5) -> torch.Tensor:
    """
    Computes inverse-frequency class weights strictly from the training split
    to mitigate class imbalance inside nn.CrossEntropyLoss(weight=...).
    """
    df = pd.read_csv(train_csv_path)
    counts = df["label"].value_counts().reindex(range(num_classes), fill_value=1)
    total = counts.sum()
    weights = total / (num_classes * counts.to_numpy(dtype=float))
    return torch.tensor(weights, dtype=torch.float32)
