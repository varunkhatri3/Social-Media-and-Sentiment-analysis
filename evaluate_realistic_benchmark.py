from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from predict_sentiment import predict_sentiment


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

BENCHMARK_CSV = DATA_DIR / "realistic_benchmark.csv"
REPORT_PATH = DATA_DIR / "realistic_classification_report.txt"
RESULTS_PATH = DATA_DIR / "realistic_benchmark_predictions.csv"
CONFUSION_CSV = DATA_DIR / "realistic_confusion_matrix.csv"
CONFUSION_PNG = DATA_DIR / "realistic_confusion_matrix.png"

LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_TO_TARGET = {"negative": 0, "neutral": 1, "positive": 2}


def main() -> None:
    if not BENCHMARK_CSV.is_file():
        raise FileNotFoundError(f"Missing benchmark file: {BENCHMARK_CSV}")

    df = pd.read_csv(BENCHMARK_CSV)
    required_columns = {"text", "sentiment", "target"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark file is missing columns: {sorted(missing)}")

    predictions = [predict_sentiment(text) for text in df["text"].astype(str)]
    df["predicted_sentiment"] = predictions
    df["predicted_target"] = df["predicted_sentiment"].map(LABEL_TO_TARGET)
    df["correct"] = df["predicted_target"] == df["target"]
    df.to_csv(RESULTS_PATH, index=False)

    report = classification_report(
        df["target"],
        df["predicted_target"],
        labels=[0, 1, 2],
        target_names=LABEL_ORDER,
        digits=4,
        zero_division=0,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    cm = confusion_matrix(df["target"], df["predicted_target"], labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)
    cm_df.to_csv(CONFUSION_CSV)

    fig, ax = plt.subplots(figsize=(6, 4.8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix (Realistic Benchmark)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    plt.savefig(CONFUSION_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)

    accuracy = df["correct"].mean()
    print(report)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved predictions: {RESULTS_PATH}")
    print(f"Saved confusion matrix csv: {CONFUSION_CSV}")
    print(f"Saved confusion matrix image: {CONFUSION_PNG}")


if __name__ == "__main__":
    main()
