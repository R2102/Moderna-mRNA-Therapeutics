from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.module4_report_generator import (
    REQUIRED_DISCLAIMER,
    StructuredVaccineReport,
    build_report_prompt,
    generate_structured_report,
)


class Module4ReportGeneratorTests(unittest.TestCase):
    @staticmethod
    def _risk_output() -> dict[str, object]:
        return {"risk_score": 0.27, "best_model_name": "random_forest", "predicted_class": 0}

    @staticmethod
    def _retrieved_chunks() -> list[str]:
        return [
            "Guideline excerpt: Standard candidate dose is 2.5 mg for low-risk adults.",
            "Guideline excerpt: Maximum safe dose for this profile is 3.0 mg.",
            "Guideline excerpt: Monitor adverse effects and document rationale.",
        ]

    @staticmethod
    def _valid_llm_response(dose: float = 2.5) -> dict[str, object]:
        return {
            "Eligible_For_Vaccine": True,
            "Recommended_Dose_mg": dose,
            "Reasoning_Steps": [
                "Used the model risk output.",
                "Mapped risk to retrieved guideline recommendations.",
            ],
            "FDA_Safety_Warnings_Cited": ["Observe post-dose adverse-event monitoring guidance."],
            "Disclaimer": REQUIRED_DISCLAIMER,
        }

    def test_schema_missing_or_mistyped_field_raises(self):
        with self.assertRaises(ValidationError):
            StructuredVaccineReport(
                Eligible_For_Vaccine=True,
                Recommended_Dose_mg=2.0,
                Reasoning_Steps=["ok"],
                FDA_Safety_Warnings_Cited=["warn"],
            )

        with self.assertRaises(ValidationError):
            StructuredVaccineReport(
                Eligible_For_Vaccine="yes",
                Recommended_Dose_mg="2mg",
                Reasoning_Steps=["ok"],
                FDA_Safety_Warnings_Cited=["warn"],
                Disclaimer=REQUIRED_DISCLAIMER,
            )

    def test_prompt_uses_only_module1_and_module3_grounding_payloads(self):
        risk = self._risk_output()
        chunks = self._retrieved_chunks()
        prompt = build_report_prompt(risk, chunks)

        self.assertIn('"risk_score": 0.27', prompt)
        self.assertIn("Maximum safe dose for this profile is 3.0 mg.", prompt)
        self.assertNotIn("External trial fact not provided", prompt)

    def test_validation_failure_triggers_exactly_one_retry_then_succeeds(self):
        calls = {"count": 0}

        def llm(prompt: str):
            calls["count"] += 1
            if calls["count"] == 1:
                return {"not_schema": True}
            return self._valid_llm_response()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = generate_structured_report(
                module1_risk_output=self._risk_output(),
                retrieved_chunks=self._retrieved_chunks(),
                llm_callable=llm,
                output_dir=Path(tmp_dir),
                filename_prefix="retry_success",
            )

        self.assertEqual(calls["count"], 2)
        self.assertEqual(result["retries_used"], 1)

    def test_above_max_safe_dose_forces_ineligible(self):
        def llm(prompt: str):
            return self._valid_llm_response(dose=4.4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = generate_structured_report(
                module1_risk_output=self._risk_output(),
                retrieved_chunks=self._retrieved_chunks(),
                llm_callable=llm,
                output_dir=Path(tmp_dir),
                filename_prefix="dose_guardrail",
            )

            report = result["report"]
            self.assertFalse(report["Eligible_For_Vaccine"])

    def test_validation_failure_after_retry_raises_clear_error(self):
        def llm(prompt: str):
            return {"invalid_payload": "still invalid"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError) as ctx:
                generate_structured_report(
                    module1_risk_output=self._risk_output(),
                    retrieved_chunks=self._retrieved_chunks(),
                    llm_callable=llm,
                    output_dir=Path(tmp_dir),
                    filename_prefix="retry_failure",
                )
        self.assertIn("after one retry", str(ctx.exception))

    def test_disclaimer_present_for_five_reports_and_files_persisted(self):
        def llm(prompt: str):
            return self._valid_llm_response(dose=2.2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_dir = Path(tmp_dir)
            for index in range(5):
                result = generate_structured_report(
                    module1_risk_output=self._risk_output(),
                    retrieved_chunks=self._retrieved_chunks(),
                    llm_callable=llm,
                    output_dir=report_dir,
                    filename_prefix=f"sample_{index}",
                )

                report = result["report"]
                self.assertIn("Disclaimer", report)
                self.assertTrue(report["Disclaimer"].strip())
                self.assertEqual(report["Disclaimer"], REQUIRED_DISCLAIMER)

                self.assertTrue(Path(result["json_path"]).exists())
                self.assertTrue(Path(result["md_path"]).exists())
                self.assertEqual(Path(result["json_path"]).stem, Path(result["md_path"]).stem)


if __name__ == "__main__":
    unittest.main()
