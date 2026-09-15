from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALID_NUCLEOTIDES = {"A", "C", "G", "U"}


def _validate_sequence(sequence: str) -> str:
    if not isinstance(sequence, str) or len(sequence) == 0 or len(sequence) > 50:
        raise ValueError("Sequence must be a string with length between 1 and 50.")

    invalid = sorted({char for char in sequence if char not in VALID_NUCLEOTIDES})
    if invalid:
        raise ValueError(
            f"Invalid nucleotide(s) found: {', '.join(invalid)}. Allowed: A, C, G, U."
        )
    return sequence


def encode_sequence(sequence: str, embed_dim: int = 8) -> np.ndarray:
    sequence = _validate_sequence(sequence)
    if embed_dim <= 0:
        raise ValueError("embed_dim must be a positive integer.")

    rng = np.random.default_rng(42)
    embedding_table = rng.normal(loc=0.0, scale=1.0, size=(4, embed_dim))
    index_map = {"A": 0, "C": 1, "G": 2, "U": 3}
    encoded = np.vstack([embedding_table[index_map[base]] for base in sequence])
    return encoded


def _softmax(matrix: np.ndarray) -> np.ndarray:
    stabilized = matrix - matrix.max(axis=1, keepdims=True)
    exp = np.exp(stabilized)
    return exp / exp.sum(axis=1, keepdims=True)


def self_attention(
    sequence: str,
    embed_dim: int = 8,
    projection_dim: int = 8,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    if projection_dim <= 0:
        raise ValueError("projection_dim must be a positive integer.")

    encoded = encode_sequence(sequence, embed_dim=embed_dim)
    rng = np.random.default_rng(random_seed)

    w_q = rng.normal(size=(embed_dim, projection_dim))
    w_k = rng.normal(size=(embed_dim, projection_dim))
    w_v = rng.normal(size=(embed_dim, projection_dim))

    queries = encoded @ w_q
    keys = encoded @ w_k
    values = encoded @ w_v

    scores = (queries @ keys.T) / np.sqrt(projection_dim)
    attention_weights = _softmax(scores)

    return {
        "encoded": encoded,
        "queries": queries,
        "keys": keys,
        "values": values,
        "attention_weights": attention_weights,
    }


def render_attention_heatmap(
    attention_weights: np.ndarray,
    output_path: Path | str = Path("outputs/figures/attention_heatmap.png"),
) -> Path:
    if attention_weights.ndim != 2 or attention_weights.shape[0] != attention_weights.shape[1]:
        raise ValueError("attention_weights must be a square 2D matrix.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seq_len = attention_weights.shape[0]
    positions = np.arange(1, seq_len + 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    heatmap = ax.imshow(attention_weights, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks(np.arange(seq_len))
    ax.set_xticklabels(positions)
    ax.set_yticklabels(positions)
    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("Sequence Position")
    ax.set_title("Self-Attention Weights")
    fig.colorbar(heatmap, ax=ax, label="Attention Weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
