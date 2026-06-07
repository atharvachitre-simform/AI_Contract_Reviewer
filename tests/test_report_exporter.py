from src.models.models import (
    ContractReviewState,
    ReportAssemblerOutput,
    ReviewVerdict,
    RiskLevel,
    ContractMetadata,
    ObligationItem,
    ObligationFinderOutput,
    RedFlagItem,
    RedFlagDetectorOutput,
    PlainEnglishClause,
    PlainEnglishWriterOutput,
    NegotiationPriority,
    MissingClause
)
from src.helpers.report_exporter import export_as_markdown, export_as_pdf, export_as_docx


def test_report_exporters():
    # Build a mock state
    state = ContractReviewState(
        contract_id="test-123",
        perspective="Customer",
        metadata=ContractMetadata(
            document_name="Test Agreement",
            contract_type="SaaS Agreement",
            governing_law="Delaware"
        ),
        final_report=ReportAssemblerOutput(
            verdict=ReviewVerdict.NEGOTIATE,
            overall_risk_level=RiskLevel.HIGH,
            report_summary="This is a summary of the SaaS Agreement review.",
            key_risks=["High liability caps", "Unilateral pricing changes"],
            negotiation_priorities=[
                NegotiationPriority(
                    title="Liability Parity",
                    priority=1,
                    reason="Vendor liability is uncapped while Customer liability is capped.",
                    recommended_action="Make liability caps mutual.",
                    related_clauses=["Limitation of Liability"]
                )
            ],
            missing_clauses=[
                MissingClause(
                    category="Governing Law",
                    reason="Explicit governing law section is absent.",
                    impact="Potential jurisdictional disputes."
                )
            ],
            recommended_next_steps=["Contact legal council", "Request revisions"]
        ),
        red_flag_detection=RedFlagDetectorOutput(
            red_flags=[
                RedFlagItem(
                    pattern_name="Uncapped Liability",
                    severity=RiskLevel.CRITICAL,
                    description="The agreement has uncapped vendor liability.",
                    evidence=["Section 10.1: Liability of the vendor shall be unlimited."],
                    safer_alternative="Limit liability to 12 months fees."
                )
            ]
        ),
        obligation_finding=ObligationFinderOutput(
            obligations=[
                ObligationItem(
                    party="Customer",
                    obligation="Pay invoices within 30 days.",
                    obligation_type="payment",
                    due_date="30 days",
                    frequency="monthly",
                    condition="after invoice receipt"
                )
            ]
        ),
        plain_english=PlainEnglishWriterOutput(
            executive_summary="SaaS contract review summary.",
            clause_summaries=[
                PlainEnglishClause(
                    clause_type="Payment Terms",
                    original_text="Customer shall pay within 30 days.",
                    plain_english="You have to pay in 30 days.",
                    why_it_matters="Late fees may apply.",
                    party_burden="High burden on Customer."
                )
            ]
        )
    )

    # 1. Test Markdown Export
    md_report = export_as_markdown(state)
    assert "# Contract Review Report" in md_report
    assert "**Verdict**: NEGOTIATE" in md_report
    assert "**Perspective**: Customer" in md_report
    assert "Uncapped Liability" in md_report
    assert "Pay invoices within 30 days" in md_report
    assert "Delaware" in md_report

    # 2. Test PDF Export
    pdf_bytes = export_as_pdf(state)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF")

    # 3. Test DOCX Export
    docx_bytes = export_as_docx(state)
    assert len(docx_bytes) > 0
    # docx is a zip file starting with PK zip header
    assert docx_bytes.startswith(b"PK")
