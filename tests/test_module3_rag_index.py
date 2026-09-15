from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.module3_rag_index import (
    build_guideline_index,
    query_guidelines,
    split_guideline_document,
)


TOPIC_TERMS = ["dose", "storage", "adverse", "pregnancy", "booster"]


def _fake_embedding_function(texts: list[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        normalized = text.lower()
        vector = [float(normalized.count(term)) for term in TOPIC_TERMS]
        vector.append(float(len(normalized.split())))
        vectors.append(vector)
    return np.asarray(vectors, dtype=float)


def _max_boundary_overlap(left_chunk: str, right_chunk: str) -> int:
    left_tokens = left_chunk.split()
    right_tokens = right_chunk.split()
    max_window = min(len(left_tokens), len(right_tokens), 120)
    for overlap in range(max_window, 0, -1):
        if left_tokens[-overlap:] == right_tokens[:overlap]:
            return overlap
    return 0


class Module3RagIndexTests(unittest.TestCase):
    @staticmethod
    def _write_guideline_file(path: Path, tokens_count: int = 1800) -> Path:
        repeated_terms = TOPIC_TERMS * ((tokens_count // len(TOPIC_TERMS)) + 1)
        content = " ".join(repeated_terms[:tokens_count])
        path.write_text(content, encoding="utf-8")
        return path

    def test_chunking_range_and_overlap(self):
        text = " ".join([f"token{i}" for i in range(1800)])
        chunks = split_guideline_document(text)
        lengths = [len(chunk.split()) for chunk in chunks]

        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(300 <= length <= 500 for length in lengths))

        overlaps = [
            _max_boundary_overlap(chunks[index], chunks[index + 1])
            for index in range(len(chunks) - 1)
        ]
        self.assertTrue(all(overlap >= 30 for overlap in overlaps))

    def test_embedding_count_matches_chunk_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            guideline_path = self._write_guideline_file(Path(tmp_dir) / "guidelines.txt")
            result = build_guideline_index(
                guideline_path=guideline_path,
                persist_directory=Path(tmp_dir) / "chroma",
                collection_name="week3_test_count",
                embedding_function=_fake_embedding_function,
            )
            self.assertEqual(result["embedding_count"], len(result["chunks"]))

    def test_persistence_and_query_top_k_with_latency(self):
        questions = [
            "What is the recommended dose schedule?",
            "How should the formulation be kept in storage?",
            "Which adverse events are highlighted?",
            "What are the pregnancy safety notes?",
            "When should a booster be considered?",
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            guideline_path = self._write_guideline_file(Path(tmp_dir) / "guidelines.txt")
            persist_directory = Path(tmp_dir) / "chroma"
            collection_name = "week3_query_test"

            build_guideline_index(
                guideline_path=guideline_path,
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding_function=_fake_embedding_function,
            )

            for question in questions:
                result = query_guidelines(
                    question=question,
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                    top_k=3,
                    embedding_function=_fake_embedding_function,
                )
                self.assertEqual(len(result["chunks"]), 3)
                self.assertTrue(all(isinstance(chunk, str) for chunk in result["chunks"]))
                self.assertEqual(result["distances"], sorted(result["distances"]))
                self.assertGreater(result["latency_seconds"], 0.0)
                self.assertLess(result["latency_seconds"], 5.0)


if __name__ == "__main__":
    unittest.main()
