"""Diagnóstico offline de integridade e perda de informação entre as etapas do Deep Research.

Compara as 5 etapas:
1. Texto bruto produzido para cada lote.
2. Consolidação intermediária.
3. Texto bruto de cada seção.
4. Relatório após validação/revisão.
5. Documento final apresentado ao usuário.

Não grava logs persistentes nem expõe segredos/conversas pessoais."""

from __future__ import annotations

import re
from typing import Any

from backend.core.redact import redact_secrets
from backend.research.report import (
    BatchAnalysisRecord,
    ConsolidatedAnalysis,
    EvidenceBatch,
    ReportSectionSpec,
    ReportStructureValidation,
    Review,
    extract_report_sections,
    validate_report_structure,
)
from backend.research.state import ResearchTask

_FINDING_RE = re.compile(r"\b(F\d+)\b")
_EVIDENCE_RE = re.compile(r"\b(E\d+)\b")


def _safe_preview(text: str, limit: int = 360) -> str:
    return redact_secrets((text or "").strip())[:limit]


def build_staged_diagnostic(
    task: ResearchTask,
    batches: list[EvidenceBatch],
    batch_records: list[BatchAnalysisRecord],
    consolidated: ConsolidatedAnalysis,
    section_records: list[dict[str, Any]],
    assembled_text: str,
    validated_text: str,
    final_text: str,
    *,
    section_specs: list[ReportSectionSpec] | None = None,
    structural_validation: ReportStructureValidation | None = None,
    review: Review | None = None,
) -> dict[str, Any]:
    """Compara as 5 etapas do relatório em etapas e aponta perdas de informação entre elas."""
    led = task.ledger
    sv = structural_validation or validate_report_structure(
        task, validated_text, sections=section_specs, section_results=section_records
    )

    batches_raw = [
        {
            "batch_index": r.batch_index,
            "ok": r.ok,
            "error": redact_secrets(r.error) if r.error else None,
            "finish_reason": r.finish_reason,
            "sent_evidence_ids": list(r.evidence_ids),
            "cited_evidence_ids": list(r.cited_evidence_ids),
            "finding_ids": list(r.finding_ids),
            "raw_chars": len(r.raw_text or ""),
            "raw_preview": _safe_preview(r.raw_text),
            "extracted_finding_notes": len(r.finding_notes),
            "extracted_observations": len(r.observations),
            "extracted_conflicts": len(r.conflicts),
            "extracted_gaps": len(r.gaps),
            "dropped_lines": r.dropped_lines,
        }
        for r in batch_records
    ]

    consolidation_stage = {
        "collected_evidence_ids": list(consolidated.collected_evidence_ids),
        "sent_evidence_ids": list(consolidated.sent_evidence_ids),
        "analyzed_evidence_ids": list(consolidated.analyzed_evidence_ids),
        "cited_in_batches_ids": list(consolidated.cited_in_batches_ids),
        "used_in_findings_ids": list(consolidated.used_in_findings_ids),
        "unreviewed_evidence_ids": list(consolidated.unreviewed_evidence_ids),
        "finding_notes_keys": sorted(consolidated.finding_notes.keys()),
        "observations_count": len(consolidated.observations),
        "conflicts_count": len(consolidated.conflicts),
        "gaps_count": len(consolidated.gaps),
        "failed_batches": list(consolidated.failed_batches),
    }

    sections_raw = [
        {
            "index": sr.get("index"),
            "key": sr.get("key"),
            "label": sr.get("label"),
            "finish_reason": sr.get("finish_reason"),
            "recovered": bool(sr.get("recovered")),
            "recovery_failed": bool(sr.get("recovery_failed")),
            "empty": bool(sr.get("empty")),
            "raw_chars": len(sr.get("raw_text") or ""),
            "raw_preview": _safe_preview(sr.get("raw_text") or ""),
            "cited_findings": sorted(set(_FINDING_RE.findall(sr.get("raw_text") or ""))),
            "cited_evidences": sorted(set(_EVIDENCE_RE.findall(sr.get("raw_text") or ""))),
        }
        for sr in section_records
    ]

    post_validation = {
        "assembled_chars": len(assembled_text or ""),
        "validated_chars": len(validated_text or ""),
        "review_notes": list(review.notes) if review else [],
        "review_hypotheses_count": len(review.hypotheses) if review else 0,
        "review_removed_cves": review.removed if review else 0,
        "review_invented_findings": review.invented if review else 0,
        "review_invalid_refs": review.invalid_refs if review else 0,
        "structural_valid": sv.valid,
        "empty_sections": list(sv.empty_sections),
        "missing_sections": list(sv.missing_sections),
        "truncated_sections": list(sv.truncated_sections),
        "structural_issues": list(sv.issues),
    }

    final_sections = extract_report_sections(final_text)
    final_stage = {
        "final_chars": len(final_text or ""),
        "sections_present": list(final_sections.keys()),
        "non_empty_sections": [k for k, v in final_sections.items() if v.strip()],
        "empty_sections": [k for k, v in final_sections.items() if not v.strip()],
        "cited_findings": sorted(set(_FINDING_RE.findall(final_text or ""))),
        "cited_evidences": sorted(set(_EVIDENCE_RE.findall(final_text or ""))),
    }

    ledger_fids = {f.id for f in led.findings}
    raw_sec_fids = {fid for sr in sections_raw for fid in sr["cited_findings"]} & ledger_fids
    assembled_fids = set(_FINDING_RE.findall(assembled_text or "")) & ledger_fids
    validated_fids = set(_FINDING_RE.findall(validated_text or "")) & ledger_fids
    final_fids = set(final_stage["cited_findings"]) & ledger_fids

    losses = {
        "evidence_collected_vs_sent": sorted(set(consolidated.collected_evidence_ids) - set(consolidated.sent_evidence_ids)),
        "evidence_sent_vs_processed": sorted(set(consolidated.sent_evidence_ids) - set(consolidated.analyzed_evidence_ids)),
        "evidence_processed_vs_used_in_findings": sorted(set(consolidated.analyzed_evidence_ids) - set(consolidated.used_in_findings_ids)),
        "findings_ledger_vs_sections_raw": sorted(ledger_fids - raw_sec_fids),
        "findings_sections_raw_vs_validated": sorted(raw_sec_fids - validated_fids),
        "findings_assembled_vs_validated": sorted(assembled_fids - validated_fids),
        "findings_ledger_vs_final": sorted(ledger_fids - final_fids),
        "truncated_batches": [r["batch_index"] for r in batches_raw if r["finish_reason"] == "length"],
        "truncated_sections": list(sv.truncated_sections),
        "empty_or_missing_sections": sorted(set(sv.empty_sections + sv.missing_sections)),
        "chars_dropped_in_validation": max(0, len(assembled_text or "") - len(validated_text or "")),
    }

    return {
        "batches_planned": len(batches),
        "stages": {
            "1_batches_raw": batches_raw,
            "2_consolidation": consolidation_stage,
            "3_sections_raw": sections_raw,
            "4_post_validation": post_validation,
            "5_final_document": final_stage,
        },
        "information_losses": losses,
    }
