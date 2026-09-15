from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from src.module2_attention import (
    encode_sequence,
    render_attention_heatmap,
    self_attention,
)


class Module2AttentionTests(unittest.TestCase):
    def test_encode_sequence_returns_expected_shape(self):
        encoded = encode_sequence("ACGU", embed_dim=12)
        self.assertEqual(encoded.shape, (4, 12))

    def test_encode_sequence_rejects_invalid_nucleotide(self):
        with self.assertRaises(ValueError):
            encode_sequence("ACGX")

    def test_self_attention_returns_square_weights_matrix(self):
        sequence = "ACGUACGUACGUACG"
        result = self_attention(sequence, embed_dim=10, projection_dim=6, random_seed=7)
        weights = result["attention_weights"]
        self.assertEqual(weights.shape, (15, 15))

    def test_attention_rows_sum_to_one(self):
        sequence = "ACGUACGUACGUACG"
        weights = self_attention(sequence)["attention_weights"]
        self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))

    def test_heatmap_is_saved_and_openable(self):
        sequence = "ACGUACGUACGUACG"
        weights = self_attention(sequence)["attention_weights"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "outputs" / "figures" / "attention_heatmap.png"
            written = render_attention_heatmap(weights, output_path=output_path)

            self.assertTrue(written.exists())
            self.assertGreater(written.stat().st_size, 0)

            image = mpimg.imread(written)
            self.assertGreater(image.size, 0)


if __name__ == "__main__":
    unittest.main()
