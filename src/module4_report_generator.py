from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, ValidationError


REQUIRED_DISCLAIMER = (
    "Safety Notice: This report is decision support only and must be reviewed by a "
    "licensed clinician before any treatment decision."
)


class StructuredVaccineReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Eligible_For_Vaccine: bool
    Recommended_Dose_mg: float
    Reasoning_Steps: list[str]
    FDA_Safety_Warnings_Cited: list[str]
    Disclaimer: str


def build_report_prompt(
    module1_risk_output: dict[str, object],
    retrieved_chunks: list[str],
) -> str:
    if not isinstance(module1_risk_output, dict) or not module1_risk_output:
        raise ValueError("module1_risk_output must be a non-empty dictionary.")
    if not isinstance(retrieved_chunks, list) or not retrieved_chunks:
        raise ValueError("retrieved_chunks must be a non-empty list of strings.")
    if not all(isinstance(chunk, str) and chunk.strip() for chunk in retrieved_chunks):
        raise ValueError("retrieved_chunks must contain only non-empty strings.")

    risk_payload = json.dumps(module1_risk_output, indent=2, sort_keys=True)
    retrieval_payload = json.dumps(retrieved_chunks, indent=2)

    return (
        "Generate a structured clinical decision report using only the factual inputs below.\n\n"
        "Module 1 Risk Output (factual grounding):\n"
        f"{risk_payload}\n\n"
        "Module 3 Retrieved Guideline Chunks (factual grounding):\n"
        f"{retrieval_payload}\n\n"
        "Return a JSON object with exactly these fields:\n"
        "- Eligible_For_Vaccine (bool)\n"
        "- Recommended_Dose_mg (float)\n"
        "- Reasoning_Steps (List[str])\n"
        "- FDA_Safety_Warnings_Cited (List[str])\n"
        "- Disclaimer (str)\n"
        f"The Disclaimer must be exactly: {REQUIRED_DISCLAIMER}"
    )


def _extract_max_safe_dose_mg(retrieved_chunks: list[str]) -> float | None:
    patterns = [
        r"maximum safe dose[^0-9]{0,40}(\d+(?:\.\d+)?)\s*mg",
        r"max(?:imum)? dose[^0-9]{0,40}(\d+(?:\.\d+)?)\s*mg",
        r"must not exceed[^0-9]{0,40}(\d+(?:\.\d+)?)\s*mg",
        r"do not exceed[^0-9]{0,40}(\d+(?:\.\d+)?)\s*mg",
    ]
    extracted: list[float] = []
    for chunk in retrieved_chunks:
        lowered = chunk.lower()
        for pattern in patterns:
            for match in re.findall(pattern, lowered):
                extracted.append(float(match))
    if not extracted:
        return None
    return min(extracted)


def _coerce_llm_response(raw_response: dict[str, object] | str) -> dict[str, object]:
    if isinstance(raw_response, dict):
        return raw_response
    if isinstance(raw_response, str):
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response is not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON must be an object.")
        return parsed
    raise ValueError("LLM response must be a dict or JSON string.")


def _enforce_guideline_dose_limit(
    report: StructuredVaccineReport, retrieved_chunks: list[str]
) -> StructuredVaccineReport:
    max_safe_dose = _extract_max_safe_dose_mg(retrieved_chunks)
    if max_safe_dose is None:
        return report.model_copy(update={"Disclaimer": REQUIRED_DISCLAIMER})

    if report.Recommended_Dose_mg <= max_safe_dose:
        return report.model_copy(update={"Disclaimer": REQUIRED_DISCLAIMER})

    updated_steps = list(report.Reasoning_Steps) + [
        (
            f"Candidate dose rejected because it exceeds guideline maximum safe dose "
            f"of {max_safe_dose} mg."
        )
    ]
    updated_warnings = list(report.FDA_Safety_Warnings_Cited)
    updated_warnings.append(f"Maximum safe dose exceeded: {max_safe_dose} mg")

    return report.model_copy(
        update={
            "Eligible_For_Vaccine": False,
            "Reasoning_Steps": updated_steps,
            "FDA_Safety_Warnings_Cited": updated_warnings,
            "Disclaimer": REQUIRED_DISCLAIMER,
        }
    )


def _default_report_filename_prefix() -> str:
    return datetime.now(tz=timezone.utc).strftime("report_%Y%m%dT%H%M%S%fZ")


def persist_report(
    report: StructuredVaccineReport,
    output_dir: Path | str = Path("outputs/reports"),
    filename_prefix: str | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or _default_report_filename_prefix()

    json_path = output_dir / f"{prefix}.json"
    md_path = output_dir / f"{prefix}.md"

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Structured Vaccine Report",
                "",
                f"- **Eligible_For_Vaccine:** {report.Eligible_For_Vaccine}",
                f"- **Recommended_Dose_mg:** {report.Recommended_Dose_mg}",
                "",
                "## Reasoning Steps",
                *[f"- {step}" for step in report.Reasoning_Steps],
                "",
                "## FDA Safety Warnings Cited",
                *[f"- {warning}" for warning in report.FDA_Safety_Warnings_Cited],
                "",
                "## Disclaimer",
                report.Disclaimer,
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {"json_path": json_path, "md_path": md_path}


def generate_structured_report(
    module1_risk_output: dict[str, object],
    retrieved_chunks: list[str],
    llm_callable: Callable[[str], dict[str, object] | str],
    output_dir: Path | str = Path("outputs/reports"),
    filename_prefix: str | None = None,
) -> dict[str, object]:
    prompt = build_report_prompt(module1_risk_output, retrieved_chunks)

    last_error: Exception | None = None
    validated_report: StructuredVaccineReport | None = None
    retries = 0

    for attempt in range(2):
        try:
            raw_response = llm_callable(prompt)
            parsed = _coerce_llm_response(raw_response)
            validated_report = StructuredVaccineReport.model_validate(parsed)
            break
        except (ValidationError, ValueError) as exc:
            last_error = exc
            if attempt == 0:
                retries = 1
                continue
            raise ValueError(
                "LLM response failed schema validation after one retry."
            ) from exc

    if validated_report is None:
        raise ValueError("Unable to generate a validated report.") from last_error

    enforced_report = _enforce_guideline_dose_limit(validated_report, retrieved_chunks)
    persisted_paths = persist_report(
        enforced_report, output_dir=output_dir, filename_prefix=filename_prefix
    )

    return {
        "prompt": prompt,
        "report": enforced_report.model_dump(),
        "retries_used": retries,
        "json_path": persisted_paths["json_path"],
        "md_path": persisted_paths["md_path"],
    }
