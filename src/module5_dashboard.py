from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_roc_curves(outputs_dir: Path) -> list[tuple[str, pd.DataFrame]]:
    curves = []
    for path in sorted(outputs_dir.glob("roc_curve_*.csv")):
        curve = pd.read_csv(path)
        required_columns = {"fpr", "tpr"}
        if not required_columns.issubset(curve.columns):
            raise ValueError(f"ROC artifact is missing required columns: {path}")
        curves.append((path.stem.removeprefix("roc_curve_"), curve))

    if not curves:
        raise FileNotFoundError(f"No ROC curve artifacts found in {outputs_dir}")
    return curves


def _load_attention_image(attention_path: Path) -> np.ndarray:
    if not attention_path.exists():
        raise FileNotFoundError(f"Attention heatmap not found: {attention_path}")
    image = mpimg.imread(attention_path)
    if image.size == 0:
        raise ValueError(f"Attention heatmap is empty: {attention_path}")
    return image


def _load_latencies(latency_log_path: Path) -> list[tuple[str, float]]:
    if not latency_log_path.exists():
        raise FileNotFoundError(f"Latency log not found: {latency_log_path}")

    latencies = []
    for line in latency_log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        latencies.append((str(entry["question"]), float(entry["latency_seconds"])))

    if not latencies:
        raise ValueError(f"Latency log has no entries: {latency_log_path}")
    return latencies


def generate_dashboard(
    outputs_dir: Path | str = Path("outputs"),
    output_path: Path | str | None = None,
    attention_path: Path | str | None = None,
    latency_log_path: Path | str | None = None,
) -> Path:
    """Compose the saved Module 1, 2, and 3 results into one dashboard image."""
    outputs_dir = Path(outputs_dir)
    output_path = Path(output_path or outputs_dir / "figures" / "dashboard.png")
    attention_path = Path(
        attention_path or outputs_dir / "figures" / "attention_heatmap.png"
    )
    latency_log_path = Path(
        latency_log_path or outputs_dir / "chroma" / "query_latency_log.jsonl"
    )

    roc_curves = _load_roc_curves(outputs_dir)
    attention_image = _load_attention_image(attention_path)
    latencies = _load_latencies(latency_log_path)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    roc_axis, attention_axis, latency_axis = axes
    for model_name, curve in roc_curves:
        auc_value = curve["auc"].iloc[0] if "auc" in curve else None
        label = model_name if auc_value is None else f"{model_name} (AUC={auc_value:.3f})"
        roc_axis.plot(curve["fpr"], curve["tpr"], label=label)
    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    roc_axis.set_title("ML ROC Curve")
    roc_axis.set_xlabel("False Positive Rate")
    roc_axis.set_ylabel("True Positive Rate")
    roc_axis.legend(fontsize="small")

    attention_axis.imshow(attention_image)
    attention_axis.set_title("Attention Heatmap")
    attention_axis.set_xlabel("Sequence Position")
    attention_axis.set_ylabel("Sequence Position")

    labels = [
        question if len(question) <= 24 else question[:21] + "..."
        for question, _ in latencies
    ]
    latency_axis.bar(labels, [latency for _, latency in latencies], color="tab:orange")
    latency_axis.set_title("Query Latency Comparison")
    latency_axis.set_ylabel("Latency (seconds)")
    latency_axis.tick_params(axis="x", labelrotation=45)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


run_dashboard = generate_dashboard


if __name__ == "__main__":
    generate_dashboard()