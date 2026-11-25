import torch
import torch.nn as nn
from transformers import AutoModel
from .dataset import MODEL_NAME, MODEL_REVISION, get_tokenizer, validate_text_input


class DistilBertEmotionClassifier(nn.Module):
    """
    Hybrid Transformer + MLP classifier for 5-class emotion recognition.
    Supports 'frozen' (feature extraction) and 'top2' (partial fine-tuning ablation).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        revision: str = MODEL_REVISION,
        num_classes: int = 5,
        hidden_dim: int = 128,
        dropout_rate: float = 0.2,
        freeze_mode: str = "frozen",
    ):
        super().__init__()
        if freeze_mode not in {"frozen", "top2"}:
            raise ValueError("freeze_mode must be 'frozen' or 'top2'")

        self.freeze_mode = freeze_mode
        self.encoder = AutoModel.from_pretrained(model_name, revision=revision)

        # 1. Freeze all pretrained backbone weights by default
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 2. Controlled Ablation: Unfreeze top 2 Transformer layers (layers 4 & 5)
        if freeze_mode == "top2":
            for layer_idx in [4,5]:
                for param in self.encoder.transformer.layer[layer_idx].parameters():
                    param.requires_grad = True

        # 3. Custom PyTorch Classification Head operating on [CLS] representation (d=768)
        encoder_dim = self.encoder.config.hidden_size  # 768 for distilbert-base-uncased
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Extract [CLS] token hidden state at sequence index 0 -> Shape: (batch_size, 768)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)  # Shape: (batch_size, num_classes)
        return logits

    def parameter_summary(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "freeze_mode": self.freeze_mode,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_pct": round((trainable_params / total_params) * 100, 2),
        }


if __name__ == "__main__":
    # Week 2 Milestone Sanity Check: Verify parameter counts, shapes, and boundary inputs
    tokenizer = get_tokenizer()

    for mode in ["frozen", "top2"]:
        model = DistilBertEmotionClassifier(freeze_mode=mode)
        print(f"Mode '{mode}' Parameter Summary:", model.parameter_summary())

    # Test batch including edge cases (empty string, emoji-only, normal text)
    sample_texts = ["", "   ", "😂😂😂", "I am terrified of failing this exam!"]
    cleaned = [validate_text_input(t)[0] for t in sample_texts]
    encoded = tokenizer(cleaned, padding="max_length", truncation=True, max_length=64, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        logits = model(encoded["input_ids"], encoded["attention_mask"])

    assert logits.shape == (4, 5), f"Expected shape (4, 5), got {logits.shape}"
    print(f"Sanity check passed! Output logits shape: {tuple(logits.shape)}")
