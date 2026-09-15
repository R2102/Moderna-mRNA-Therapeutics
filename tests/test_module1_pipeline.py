from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.module1_pipeline import (
    MissingRequiredColumnError,
    clean_data,
    load_and_validate_data,
    run_pipeline,
    split_and_scale_data,
)


class Module1PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.csv_path = cls.repo_root / "data" / "patient_records.csv"

    def test_missing_column_raises_named_error(self):
        frame = pd.read_csv(self.csv_path).drop(columns=["Age"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "missing_column.csv"
            frame.to_csv(path, index=False)
            with self.assertRaises(MissingRequiredColumnError) as ctx:
                load_and_validate_data(path)
        self.assertIn("Age", str(ctx.exception))

    def test_clean_data_has_no_null_values(self):
        raw = load_and_validate_data(self.csv_path)
        cleaned = clean_data(raw)
        self.assertEqual(int(cleaned.isnull().sum().sum()), 0)

    def test_scaling_is_zero_mean_and_unit_variance_on_training_set(self):
        raw = load_and_validate_data(self.csv_path)
        cleaned = clean_data(raw)
        split_data = split_and_scale_data(cleaned, random_seed=17)
        train = split_data["x_train"]

        means = train[["Age", "Biomarker_Level", "Vaccine_Dose_mcg"]].mean().values
        stds = train[["Age", "Biomarker_Level", "Vaccine_Dose_mcg"]].std(ddof=0).values

        self.assertTrue(np.all(np.isclose(means, 0, atol=1e-8)))
        self.assertTrue(np.all(np.isclose(stds, 1, atol=1e-8)))

    def test_split_reproducibility_with_seed(self):
        raw = load_and_validate_data(self.csv_path)
        cleaned = clean_data(raw)

        split_one = split_and_scale_data(cleaned, random_seed=99)
        split_two = split_and_scale_data(cleaned, random_seed=99)

        self.assertTrue(split_one["x_train"].index.equals(split_two["x_train"].index))
        self.assertTrue(split_one["x_test"].index.equals(split_two["x_test"].index))

    def test_models_artifacts_and_serialized_model_predictions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_pipeline(self.csv_path, Path(tmp_dir), random_seed=23)

            self.assertIn("logistic_regression", result["models"])
            self.assertIn("random_forest", result["models"])

            for model_name in ["logistic_regression", "random_forest"]:
                report_path = Path(tmp_dir) / f"classification_report_{model_name}.txt"
                matrix_path = Path(tmp_dir) / f"confusion_matrix_{model_name}.csv"
                roc_path = Path(tmp_dir) / f"roc_curve_{model_name}.csv"
                self.assertTrue(report_path.exists())
                self.assertTrue(matrix_path.exists())
                self.assertTrue(roc_path.exists())
                self.assertTrue(len(report_path.read_text().strip()) > 0)
                matrix = pd.read_csv(matrix_path)
                roc = pd.read_csv(roc_path)
                self.assertGreater(matrix.shape[0], 0)
                self.assertIn("fpr", roc.columns)
                self.assertIn("tpr", roc.columns)

            model_path = Path(tmp_dir) / "risk_model.joblib"
            self.assertTrue(model_path.exists())

            loaded_bundle = joblib.load(model_path)
            x_test_unscaled = result["split_data"]["x_test"].copy()
            x_test_unscaled[["Age", "Biomarker_Level", "Vaccine_Dose_mcg"]] = (
                result["split_data"]["scaler"].inverse_transform(
                    x_test_unscaled[["Age", "Biomarker_Level", "Vaccine_Dose_mcg"]]
                )
            )

            original_predictions = result["models"][result["best_model_name"]].predict(
                result["split_data"]["x_test"]
            )
            loaded_predictions = loaded_bundle.predict(x_test_unscaled)
            self.assertTrue(np.array_equal(original_predictions, loaded_predictions))


if __name__ == "__main__":
    unittest.main()
