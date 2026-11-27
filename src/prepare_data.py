import json
import os
from collections import Counter
from datasets import load_dataset
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"


# 1. Configuration & Output Paths
DATASET_NAME = "google-research-datasets/go_emotions"
CONFIG_NAME = "simplified"
SUMMARY_FILE = os.path.join(DATA_DIR, "dataset_summary.txt")

TARGET_NAMES = ["Joy", "Anger", "Sadness", "Fear", "Neutral"]

# Deterministic mapping from GoEmotions 'simplified' label IDs (0-27) to 5 target classes
RAW_ID_TO_TARGET = {
    # Joy (0)
    17: 0,  # joy
    1: 0,   # amusement
    13: 0,  # excitement
    20: 0,  # optimism
    21: 0,  # pride
    23: 0,  # relief
    # Anger (1)
    2: 1,   # anger
    3: 1,   # annoyance
    10: 1,  # disapproval
    11: 1,  # disgust
    # Sadness (2)
    25: 2,  # sadness
    9: 2,   # disappointment
    12: 2,  # embarrassment
    16: 2,  # grief
    24: 2,  # remorse
    # Fear (3)
    14: 3,  # fear
    19: 3,  # nervousness
    # Neutral (4)
    27: 4,  # neutral
}


def resolve_single_label(raw_labels: list[int]) -> int | None:
    """
    Deterministic single-label resolution rule:
    - Empty label list -> None (discard)
    - Neutral (27) is retained ONLY if it is the sole label ([27]).
    - Any unmapped label ID -> None (discard) so secondary states don't confound.
    - Multiple raw labels are retained ONLY if all map to the exact same target class.
    """
    if not raw_labels:
        return None

    if 27 in raw_labels:
        return 4 if raw_labels == [27] else None

    mapped = []
    for lbl in raw_labels:
        if lbl not in RAW_ID_TO_TARGET:
            return None  # Discard if mixed with unmapped labels
        mapped.append(RAW_ID_TO_TARGET[lbl])

    unique_targets = set(mapped)
    if len(unique_targets) == 1:
        return unique_targets.pop()

    return None  # Discard cross-category multi-label comments


def process_split(split_dataset) -> tuple[pd.DataFrame, dict]:
    records = []
    stats = Counter()

    for row in split_dataset:
        stats["total_raw"] += 1
        text = row["text"]
        if not text or not text.strip():
            stats["discarded_empty"] += 1
            continue

        target = resolve_single_label(row["labels"])
        if target is None:
            stats["discarded_multilabel_or_unmapped"] += 1
            continue

        stats["retained"] += 1
        records.append({
            "id": row["id"],
            "text": text.strip(),
            "label": target,
            "label_name": TARGET_NAMES[target],
            "raw_labels": json.dumps(row["labels"]),
        })

    return pd.DataFrame(records), dict(stats)


def format_mapping_section(raw_label_names: list[str]) -> list[str]:
    """Formats the exact RAW_ID_TO_TARGET dictionary and excluded IDs for documentation."""
    lines = ["--- 1. DETERMINISTIC LABEL MAPPING (RAW_ID_TO_TARGET) ---"]

    for target_idx, target_name in enumerate(TARGET_NAMES):
        mapped_items = [
            f"{raw_id} ({raw_label_names[raw_id]})"
            for raw_id, tgt in sorted(RAW_ID_TO_TARGET.items())
            if tgt == target_idx
        ]
        lines.append(f"  Target {target_idx} [{target_name:<7}] <- {', '.join(mapped_items)}")

    excluded_items = [
        f"{raw_id} ({name})"
        for raw_id, name in enumerate(raw_label_names)
        if raw_id not in RAW_ID_TO_TARGET
    ]
    lines.append(f"  Excluded [Unmapped] <- {', '.join(excluded_items)}")
    return lines


def build_summary_report(
    raw_label_names: list[str],
    retention_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    pct_df: pd.DataFrame,
) -> str:
    lines = [
        "======================================================================",
        "GOEMOTIONS DETERMINISTIC 5-CLASS PREPROCESSING SUMMARY",
        "======================================================================",
        f"Source Dataset : {DATASET_NAME} (config: '{CONFIG_NAME}')",
        f"Output Folder  : {DATA_DIR}/",
        "Resolution Rule: Within-group multi-labels retained; cross-group/unmapped",
        "                 discarded; Neutral (27) retained iff sole label ([27]).",
        "",
        *format_mapping_section(raw_label_names),
        "",
        "--- 2. SPLIT RETENTION & FILTERING BREAKDOWN ---",
        retention_df.to_string(index=False),
        "",
        "--- 3. CLASS COUNTS PER SPLIT ---",
        counts_df.to_string(index=False),
        "",
        "--- 4. CLASS DISTRIBUTION PERCENTAGES (SHOWING IMBALANCE) ---",
        pct_df.to_string(index=False),
        "",
        "Note: Aggregating fine-grained labels does not balance class frequencies.",
        "Validation and test splits remain unresampled to reflect natural priors;",
        "training imbalance is handled via inverse-frequency class weights.",
        "======================================================================",
    ]
    return "\n".join(lines)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    ds = load_dataset(DATASET_NAME, CONFIG_NAME)
    raw_label_names = ds["train"].features["labels"].feature.names

    retention_rows = []
    count_rows = []
    pct_rows = []
    seen_ids = {}

    for split_name in ["train", "validation", "test"]:
        df, stats = process_split(ds[split_name])
        seen_ids[split_name] = set(df["id"])

        out_csv = os.path.join(DATA_DIR, f"data_{split_name}.csv")
        df.to_csv(out_csv, index=False)

        total_retained = stats["retained"]
        counts = df["label_name"].value_counts().to_dict()

        retention_rows.append({
            "Split": split_name,
            "Raw_Rows": stats["total_raw"],
            "Retained": total_retained,
            "Discarded_Empty": stats.get("discarded_empty", 0),
            "Discarded_Label_Conflict": stats.get("discarded_multilabel_or_unmapped", 0),
            "Retention_%": round((total_retained / stats["total_raw"]) * 100, 2),
        })

        count_rows.append({
            "Split": split_name,
            "Total": total_retained,
            **{name: counts.get(name, 0) for name in TARGET_NAMES},
        })

        pct_rows.append({
            "Split": split_name,
            **{
                name: f"{(counts.get(name, 0) / total_retained) * 100:.2f}%"
                for name in TARGET_NAMES
            },
        })

    # Verify zero ID leakage across official splits
    assert seen_ids["train"].isdisjoint(seen_ids["validation"]), "Leakage: train/val!"
    assert seen_ids["train"].isdisjoint(seen_ids["test"]), "Leakage: train/test!"
    assert seen_ids["validation"].isdisjoint(seen_ids["test"]), "Leakage: val/test!"

    retention_df = pd.DataFrame(retention_rows)
    counts_df = pd.DataFrame(count_rows)
    pct_df = pd.DataFrame(pct_rows)

    report_text = build_summary_report(raw_label_names, retention_df, counts_df, pct_df)

    print(report_text)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(f"\nSaved CSV splits and summary report to: {DATA_DIR}/")


if __name__ == "__main__":
    main()
