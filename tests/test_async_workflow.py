"""Unit tests for the async contract review workflow and checkpointer integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.models import ContractReviewState, ProcessingStatus
from src.workflows.async_workflow import AsyncContractReviewWorkflow


def test_async_workflow_run_streaming_imports_and_flow():
    """Verify that run_streaming runs without ModuleNotFoundError on step loads."""
    async def run_test():
        workflow = AsyncContractReviewWorkflow()

        # Create dummy outputs for each step matching the expected models
        from src.models import (
            ClauseExtractorOutput,
            ObligationFinderOutput,
            RedFlagDetectorOutput,
            RiskScorerOutput,
            PlainEnglishWriterOutput,
            ReportAssemblerOutput,
        )

        dummy_extractor_output = ClauseExtractorOutput(clauses=[])
        dummy_obligation_output = ObligationFinderOutput(obligations=[])
        dummy_red_flag_output = RedFlagDetectorOutput(red_flags=[])
        dummy_risk_output = RiskScorerOutput(issues=[])
        dummy_plain_output = PlainEnglishWriterOutput(executive_summary="Summary", clause_summaries=[])
        dummy_report_output = ReportAssemblerOutput(report_summary="Final Report")

        # Mock all agent functions imported dynamically in run_streaming
        mock_extract_clauses = MagicMock(return_value=dummy_extractor_output)
        mock_find_obligations = MagicMock(return_value=dummy_obligation_output)
        mock_detect_red_flags = MagicMock(return_value=dummy_red_flag_output)
        mock_score_risks = MagicMock(return_value=dummy_risk_output)
        mock_generate_plain_english = MagicMock(return_value=dummy_plain_output)
        mock_assemble_report = MagicMock(return_value=dummy_report_output)

        # Mock RedisCheckpointer to return empty completed steps and save successfully
        mock_checkpointer = MagicMock()
        mock_checkpointer.completed_steps = AsyncMock(return_value=[])
        mock_checkpointer.save = AsyncMock()
        mock_checkpointer.load = AsyncMock(return_value=None)

        # Apply patches
        patches = [
            patch("src.workflows.async_workflow.RedisCheckpointer", return_value=mock_checkpointer),
            patch("src.agents.clause_extractor.extract_clauses", mock_extract_clauses),
            patch("src.agents.obligation_finder.find_obligations", mock_find_obligations),
            patch("src.agents.red_flag_detector.detect_red_flags", mock_detect_red_flags),
            patch("src.agents.risk_scorer.score_risks", mock_score_risks),
            patch("src.agents.plain_english_writer.generate_plain_english", mock_generate_plain_english),
            patch("src.agents.report_assembler.assemble_report", mock_assemble_report),
        ]

        for p in patches:
            p.start()

        try:
            events = []
            async for event in workflow.run_streaming(
                contract_text="This is a dummy contract text",
                contract_id="test-contract-123",
                resume=False,
            ):
                events.append(event)

            assert len(events) > 0
            # Check that we completed successfully and the final event is 'done' with status 'completed'
            assert events[-1]["step"] == "done"
            assert events[-1]["status"] == "completed"
            assert "state" in events[-1]
            
            # Verify the structure of the final state
            final_state = events[-1]["state"]
            assert final_state["contract_id"] == "test-contract-123"
            assert final_state["status"] == ProcessingStatus.COMPLETED
        finally:
            for p in patches:
                p.stop()

    asyncio.run(run_test())


def test_async_workflow_resume_from_checkpoint():
    """Verify that run_streaming can resume from checkpointer load data."""
    async def run_test():
        workflow = AsyncContractReviewWorkflow()

        from src.models import (
            ClauseExtractorOutput,
            ObligationFinderOutput,
            RedFlagDetectorOutput,
            RiskScorerOutput,
            PlainEnglishWriterOutput,
            ReportAssemblerOutput,
        )

        dummy_extractor_output = ClauseExtractorOutput(clauses=[])
        dummy_obligation_output = ObligationFinderOutput(obligations=[])
        dummy_red_flag_output = RedFlagDetectorOutput(red_flags=[])
        dummy_risk_output = RiskScorerOutput(issues=[])
        dummy_plain_output = PlainEnglishWriterOutput(executive_summary="Summary", clause_summaries=[])
        dummy_report_output = ReportAssemblerOutput(report_summary="Final Report")

        # Setup checkpointer to return all steps as completed
        mock_checkpointer = MagicMock()
        mock_checkpointer.completed_steps = AsyncMock(return_value=[
            "clause_extraction",
            "obligation_finding",
            "red_flag_detection",
            "risk_scoring",
            "plain_english",
            "final_report",
        ])
        mock_checkpointer.save = AsyncMock()

        # Define side_effect for checkpointer load
        async def mock_load(step):
            if step == "clause_extraction":
                return dummy_extractor_output.model_dump()
            elif step == "obligation_finding":
                return dummy_obligation_output.model_dump()
            elif step == "red_flag_detection":
                return dummy_red_flag_output.model_dump()
            elif step == "risk_scoring":
                return dummy_risk_output.model_dump()
            elif step == "plain_english":
                return dummy_plain_output.model_dump()
            elif step == "final_report":
                return dummy_report_output.model_dump()
            return None

        mock_checkpointer.load = AsyncMock(side_effect=mock_load)

        # Since we load everything from checkpoints, agent functions should NOT be called
        mock_extract_clauses = MagicMock()

        patches = [
            patch("src.workflows.async_workflow.RedisCheckpointer", return_value=mock_checkpointer),
            patch("src.agents.clause_extractor.extract_clauses", mock_extract_clauses),
        ]

        for p in patches:
            p.start()

        try:
            events = []
            async for event in workflow.run_streaming(
                contract_text="This is a dummy contract text",
                contract_id="test-contract-123",
                resume=True,
            ):
                events.append(event)

            # All steps should be marked as skipped since they were loaded from checkpoints
            skipped_steps = [e["step"] for e in events if e["status"] == "skipped"]
            assert "clause_extraction" in skipped_steps
            assert "obligation_finding" in skipped_steps
            assert "red_flag_detection" in skipped_steps
            assert "risk_scoring" in skipped_steps
            assert "plain_english" in skipped_steps
            assert "final_report" in skipped_steps

            assert events[-1]["step"] == "done"
            assert events[-1]["status"] == "completed"
            mock_extract_clauses.assert_not_called()
        finally:
            for p in patches:
                p.stop()

    asyncio.run(run_test())
