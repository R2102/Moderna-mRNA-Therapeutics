# Moderna-mRNA-Therapeutics
Healthcare / Biotechnology — Clinical Decision Support

## Module 1: Data Pipeline & Risk Classifier

### Run
```bash
pip install -r requirements.txt
python -m src.module1_pipeline
```

### Input
The pipeline reads:
- `data/patient_records.csv`

Required columns:
- `Patient_ID`
- `Age`
- `Biomarker_Level`
- `Prior_Reaction_History`
- `Vaccine_Dose_mcg`
- `Comorbidity_Index`
- `Reaction_Score`

### Output
The pipeline writes to `outputs/`:
- `classification_report_logistic_regression.txt`
- `classification_report_random_forest.txt`
- `confusion_matrix_logistic_regression.csv`
- `confusion_matrix_random_forest.csv`
- `risk_model.joblib`

## Module 2: Self-Attention Engine

### Example usage
```python
from src.module2_attention import self_attention, render_attention_heatmap

sequence = "ACGUACGUACGUACG"
result = self_attention(sequence)
render_attention_heatmap(result["attention_weights"])
```

### Output
- `outputs/figures/attention_heatmap.png`
