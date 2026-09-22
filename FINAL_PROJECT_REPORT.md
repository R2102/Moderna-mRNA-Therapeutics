# Moderna mRNA Therapeutics — Final Project Report

## 1. Executive Summary

This project implements an end-to-end clinical decision support pipeline for mRNA vaccine safety assessment. It combines tabular risk prediction, self-attention analysis over nucleotide sequences, retrieval-augmented generation grounded in regulatory guidance, and a final structured decision report. The project demonstrates how a classical ML model, a deep learning-style attention mechanism, and a grounded retrieval pipeline can be connected into a single operational workflow.

The current implementation successfully builds the core artifacts: a trained risk model, attention heatmap, ChromaDB retrieval index, structured report schema, and the final combined dashboard image. The final dashboard is saved to `outputs/figures/dashboard.png` and uses the saved module outputs rather than hardcoded values.

## 2. Architecture Diagram

```text
Patient Records CSV
        |
        v
Module 1: Data Cleaning + ML Risk Classifier
        |
        +--> classification reports + confusion matrices + risk_model.joblib
        |
        v
Module 2: Self-Attention Engine
        |
        +--> outputs/figures/attention_heatmap.png
        |
        v
Module 3: Guideline RAG Index (ChromaDB)
        |
        +--> outputs/chroma/query_latency_log.jsonl
        |
        v
Module 4: Structured Safety Report Generator
        |
        +--> outputs/reports/*.json and *.md
        |
        v
Module 5: Final Dashboard
        |
        +--> outputs/figures/dashboard.png
```

## 3. Module 1 Results — Risk Classifier

The model pipeline loads the patient dataset, validates required columns, imputes missing values, standardizes numeric features, splits the data with a fixed random seed, and trains both logistic regression and random forest models.

The project output files currently include:

- `outputs/classification_report_logistic_regression.txt`
- `outputs/classification_report_random_forest.txt`
- `outputs/confusion_matrix_logistic_regression.csv`
- `outputs/confusion_matrix_random_forest.csv`
- `outputs/roc_curve_logistic_regression.csv`
- `outputs/roc_curve_random_forest.csv`
- `outputs/risk_model.joblib`

The best-performing model in the current run is the random forest model, with a weighted F1 score of 1.00 on the held-out test set. The generated classification report shows perfect separation on the current dataset and is saved for reuse by later modules.

## 4. Module 2 Results — Self-Attention

The attention implementation accepts a nucleotide string made of A, C, G, and U, creates an embedding matrix, computes Q, K, and V projections, and applies scaled dot-product attention. Each row of the attention matrix sums to 1.0 by construction via softmax.

The generated visualization is stored at:

- `outputs/figures/attention_heatmap.png`

This heatmap shows the strength of attention between sequence positions and allows inspection of which positions are weighted most strongly relative to others.

## 5. Module 3 Results — RAG Retrieval Index

The retrieval layer splits the guideline text into overlapping chunks, embeds the chunk text, stores the embeddings in a local ChromaDB collection, and supports natural-language similarity queries.

The persistence and query log are stored under:

- `outputs/chroma/`
- `outputs/chroma/query_latency_log.jsonl`

Latency is recorded for each query and is used for the dashboard bar chart in Module 5.

## 6. Module 4 Results — Structured Reporting

The report generator validates all model outputs against a strict Pydantic schema and persists the final output as both JSON and Markdown.

The repository includes a sample structured report in:

- `outputs/reports/sample_report.json`
- `outputs/reports/sample_report.md`

The schema enforces the required fields:

- `Eligible_For_Vaccine`
- `Recommended_Dose_mg`
- `Reasoning_Steps`
- `FDA_Safety_Warnings_Cited`
- `Disclaimer`

The project also includes the required non-clinical safety disclaimer for all generated reports.

## 7. Dashboard

The final dashboard combines the outputs from Modules 1, 2, and 3 into a single image with three panels:

1. ML ROC curve
2. Attention heatmap
3. Query latency comparison bar chart

Image output:

- `outputs/figures/dashboard.png`

The final dashboard was generated directly from the saved artifacts and saved at 200 DPI, exceeding the minimum 150 DPI requirement.

## 8. Limitations and Future Work

1. The current patient dataset is relatively small and synthetic in structure, so generalization is limited.
2. The retrieval system depends on chunk size and embedding quality; query relevance can vary with document style and vocabulary.
3. The report generation step relies on a deterministic, local mock or minimal validation path rather than a production LLM API stack.
4. The current project is a learning-oriented demo rather than a production clinical decision system.

Future improvements would include richer patient datasets, more robust retrieval tuning, retrieval quality evaluation, and a production-ready LLM integration with cloud secrets management.

## 9. Reproduction Instructions

1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run the pipeline modules in order.

```bash
PYTHONPATH=. python src/module1_pipeline.py
PYTHONPATH=. python src/module2_attention.py
PYTHONPATH=. python src/module3_rag_index.py
PYTHONPATH=. python src/module4_report_generator.py
PYTHONPATH=. python src/module5_dashboard.py
```

4. Validate the project.

```bash
PYTHONPATH=. pytest -q
```

5. Confirm final artifact locations.

- `outputs/figures/attention_heatmap.png`
- `outputs/figures/dashboard.png`
- `outputs/risk_model.joblib`
- `outputs/chroma/`
- `outputs/reports/`

## 10. Summary

This project demonstrates a practical AI engineering workflow for clinical safety support: data cleaning and risk prediction, attention-based sequence modeling, grounded retrieval over regulatory guidelines, and structured report generation. The final dashboard ties these outputs into a single view and serves as the capstone artifact for the project.
