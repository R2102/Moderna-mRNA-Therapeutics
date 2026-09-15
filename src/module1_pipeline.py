from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = [
    "Patient_ID",
    "Age",
    "Biomarker_Level",
    "Prior_Reaction_History",
    "Vaccine_Dose_mcg",
    "Comorbidity_Index",
    "Reaction_Score",
]

NUMERIC_FEATURES_TO_SCALE = ["Age", "Biomarker_Level", "Vaccine_Dose_mcg"]
TARGET_COLUMN = "Reaction_Score"


class MissingRequiredColumnError(ValueError):
    """Raised when the input CSV is missing a required column."""


@dataclass
class RiskModelBundle:
    model_name: str
    model: object
    scaler: StandardScaler
    feature_columns: list[str]

    def _prepare_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = frame.copy()
        if "Prior_Reaction_History" in features.columns:
            features = pd.get_dummies(features, columns=["Prior_Reaction_History"], drop_first=False)
        features = features.reindex(columns=self.feature_columns, fill_value=0)
        scaled = features.copy()
        scaled[NUMERIC_FEATURES_TO_SCALE] = self.scaler.transform(scaled[NUMERIC_FEATURES_TO_SCALE])
        return scaled

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        processed = self._prepare_features(frame)
        return self.model.predict(processed)


def load_and_validate_data(csv_path: Path | str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    for required_column in REQUIRED_COLUMNS:
        if required_column not in data.columns:
            raise MissingRequiredColumnError(
                f"Missing required column: {required_column}"
            )
    return data


def _fill_mode(series: pd.Series):
    mode_values = series.mode(dropna=True)
    if mode_values.empty:
        return series
    return series.fillna(mode_values.iloc[0])


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned["Biomarker_Level"] = cleaned["Biomarker_Level"].fillna(
        cleaned["Biomarker_Level"].median()
    )
    cleaned["Prior_Reaction_History"] = _fill_mode(cleaned["Prior_Reaction_History"])

    for column in cleaned.columns:
        if cleaned[column].isnull().any():
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())
            else:
                cleaned[column] = _fill_mode(cleaned[column])

    if cleaned.isnull().sum().sum() > 0:
        remaining_null_columns = cleaned.columns[cleaned.isnull().any()].tolist()
        raise ValueError(
            f"Unable to impute missing values for columns: {remaining_null_columns}"
        )

    return cleaned


def split_and_scale_data(
    cleaned_data: pd.DataFrame, random_seed: int = 42
) -> Dict[str, object]:
    features = cleaned_data.drop(columns=[TARGET_COLUMN])
    target = cleaned_data[TARGET_COLUMN]

    encoded_features = pd.get_dummies(
        features, columns=["Prior_Reaction_History"], drop_first=False
    )

    x_train, x_test, y_train, y_test = train_test_split(
        encoded_features,
        target,
        test_size=0.2,
        random_state=random_seed,
        stratify=target,
    )

    scaler = StandardScaler()
    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()

    scaler.fit(x_train_scaled[NUMERIC_FEATURES_TO_SCALE])
    x_train_scaled[NUMERIC_FEATURES_TO_SCALE] = scaler.transform(
        x_train_scaled[NUMERIC_FEATURES_TO_SCALE]
    )
    x_test_scaled[NUMERIC_FEATURES_TO_SCALE] = scaler.transform(
        x_test_scaled[NUMERIC_FEATURES_TO_SCALE]
    )

    return {
        "x_train": x_train_scaled,
        "x_test": x_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_columns": x_train_scaled.columns.tolist(),
    }


def train_models(x_train: pd.DataFrame, y_train: pd.Series, random_seed: int = 42):
    logistic_regression = LogisticRegression(random_state=random_seed, max_iter=1000)
    random_forest = RandomForestClassifier(n_estimators=200, random_state=random_seed)

    logistic_regression.fit(x_train, y_train)
    random_forest.fit(x_train, y_train)

    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
    }


def evaluate_and_save_artifacts(
    models: Dict[str, object],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    outputs_dir: Path,
) -> Dict[str, Dict[str, object]]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics: Dict[str, Dict[str, object]] = {}

    for model_name, model in models.items():
        predictions = model.predict(x_test)
        report_text = classification_report(y_test, predictions, zero_division=0)
        report_dict = classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )
        matrix = confusion_matrix(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        (outputs_dir / f"classification_report_{model_name}.txt").write_text(
            report_text
        )
        pd.DataFrame(matrix).to_csv(
            outputs_dir / f"confusion_matrix_{model_name}.csv", index=False
        )

        metrics[model_name] = {
            "f1": f1,
            "report": report_dict,
            "predictions": predictions,
        }

    return metrics


def select_and_serialize_best_model(
    models: Dict[str, object],
    metrics: Dict[str, Dict[str, object]],
    scaler: StandardScaler,
    feature_columns: list[str],
    outputs_dir: Path,
) -> Tuple[str, Path]:
    best_model_name = max(metrics.items(), key=lambda item: item[1]["f1"])[0]
    best_bundle = RiskModelBundle(
        model_name=best_model_name,
        model=models[best_model_name],
        scaler=scaler,
        feature_columns=feature_columns,
    )

    model_path = outputs_dir / "risk_model.joblib"
    joblib.dump(best_bundle, model_path)
    return best_model_name, model_path


def run_pipeline(
    csv_path: Path | str = Path("data/patient_records.csv"),
    outputs_dir: Path | str = Path("outputs"),
    random_seed: int = 42,
) -> Dict[str, object]:
    csv_path = Path(csv_path)
    outputs_dir = Path(outputs_dir)

    raw = load_and_validate_data(csv_path)
    cleaned = clean_data(raw)
    split_data = split_and_scale_data(cleaned, random_seed=random_seed)
    models = train_models(split_data["x_train"], split_data["y_train"], random_seed)
    metrics = evaluate_and_save_artifacts(
        models, split_data["x_test"], split_data["y_test"], outputs_dir
    )
    best_model_name, model_path = select_and_serialize_best_model(
        models,
        metrics,
        split_data["scaler"],
        split_data["feature_columns"],
        outputs_dir,
    )

    return {
        "cleaned_data": cleaned,
        "split_data": split_data,
        "models": models,
        "metrics": metrics,
        "best_model_name": best_model_name,
        "risk_model_path": model_path,
    }


if __name__ == "__main__":
    run_pipeline()
