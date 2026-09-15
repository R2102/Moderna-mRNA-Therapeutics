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

## Module 3: RAG Retrieval Index

### Example usage
```python
from src.module3_rag_index import build_guideline_index, query_guidelines

build_guideline_index("data/guidelines.txt")
result = query_guidelines("What are the contraindications?")
print(result["chunks"])
print(result["latency_seconds"])
```

### Output
- Local persistent ChromaDB index under `outputs/chroma/`
- Query result payload with `chunks`, `distances`, and `latency_seconds`

## Module 4: Structured Report Generator

### Example usage
```python
from src.module4_report_generator import generate_structured_report

def mock_llm(_prompt: str):
    return {
        "Eligible_For_Vaccine": True,
        "Recommended_Dose_mg": 2.5,
        "Reasoning_Steps": ["Used module outputs only."],
        "FDA_Safety_Warnings_Cited": ["Monitor adverse effects."],
        "Disclaimer": "placeholder"
    }

result = generate_structured_report(
    module1_risk_output={"risk_score": 0.27},
    retrieved_chunks=["Maximum safe dose for this profile is 3.0 mg."],
    llm_callable=mock_llm,
)
print(result["report"])
```

### Output
- Validated structured report persisted as matching files in `outputs/reports/`:
  - `*.json`
  - `*.md`
