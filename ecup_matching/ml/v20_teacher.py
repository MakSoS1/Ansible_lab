from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from .v20_strata import REASON_CODES


VERDICTS = {"MATCH", "NON_MATCH", "UNCERTAIN"}
_REQUIRED_KEYS = {
    "verdict", "reason_code", "same_product_type", "brand_left", "brand_right",
    "model_left", "model_right", "critical_attributes", "conflicts", "evidence",
}


@dataclass(frozen=True)
class TeacherDecision:
    verdict: str
    reason_code: str
    same_product_type: bool
    brand_left: str
    brand_right: str
    model_left: str
    model_right: str
    critical_attributes: dict[str, Any]
    conflicts: tuple[str, ...]
    evidence: tuple[str, ...]
    teacher_id: str
    revision: str
    prompt_sha256: str
    raw_sha256: str

    @classmethod
    def from_json(
        cls,
        raw: str,
        *,
        teacher_id: str,
        revision: str,
        prompt_sha256: str,
    ) -> "TeacherDecision":
        try:
            payload = json.loads(raw)
        except Exception as exc:
            raise ValueError("teacher output is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("teacher output must be a JSON object")
        missing = _REQUIRED_KEYS - set(payload)
        if missing:
            raise ValueError(f"teacher output missing fields: {sorted(missing)}")
        verdict = str(payload["verdict"]).strip().upper()
        reason = str(payload["reason_code"]).strip().upper()
        if verdict not in VERDICTS:
            raise ValueError(f"invalid teacher verdict: {verdict}")
        if reason not in REASON_CODES:
            raise ValueError(f"invalid teacher reason_code: {reason}")
        if not isinstance(payload["same_product_type"], bool):
            raise ValueError("same_product_type must be boolean")
        if not isinstance(payload["critical_attributes"], dict):
            raise ValueError("critical_attributes must be an object")
        if not isinstance(payload["conflicts"], list) or not isinstance(payload["evidence"], list):
            raise ValueError("conflicts/evidence must be arrays")
        if len(prompt_sha256) != 64:
            raise ValueError("prompt_sha256 must be a SHA-256 hex string")
        return cls(
            verdict=verdict,
            reason_code=reason,
            same_product_type=bool(payload["same_product_type"]),
            brand_left=str(payload["brand_left"]),
            brand_right=str(payload["brand_right"]),
            model_left=str(payload["model_left"]),
            model_right=str(payload["model_right"]),
            critical_attributes=dict(payload["critical_attributes"]),
            conflicts=tuple(str(v) for v in payload["conflicts"]),
            evidence=tuple(str(v) for v in payload["evidence"]),
            teacher_id=str(teacher_id),
            revision=str(revision),
            prompt_sha256=str(prompt_sha256).lower(),
            raw_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["conflicts"] = list(self.conflicts)
        payload["evidence"] = list(self.evidence)
        return payload


def _checker_compatible(reason: str, deterministic_reason: str) -> bool:
    d = str(deterministic_reason).upper()
    r = str(reason).upper()
    if d in {"OTHER", "SPARSE_EVIDENCE"}:
        return True
    return d == r


def consensus_label(
    first: TeacherDecision | None,
    second: TeacherDecision | None,
    *,
    deterministic_reason: str,
) -> dict[str, Any]:
    if first is None or second is None:
        return {"admitted": False, "reason": "two_teachers_required"}
    if first.teacher_id == second.teacher_id and first.revision == second.revision:
        return {"admitted": False, "reason": "independent_teachers_required"}
    if first.verdict == "UNCERTAIN" or second.verdict == "UNCERTAIN":
        return {"admitted": False, "reason": "uncertain"}
    if first.verdict != second.verdict:
        return {"admitted": False, "reason": "teacher_disagreement"}
    if first.reason_code != second.reason_code:
        return {"admitted": False, "reason": "reason_disagreement"}
    if not _checker_compatible(first.reason_code, deterministic_reason):
        return {"admitted": False, "reason": "deterministic_checker_conflict"}
    target = 1 if first.verdict == "MATCH" else 0
    provenance = {
        "teacher_ids": [first.teacher_id, second.teacher_id],
        "teacher_revisions": [first.revision, second.revision],
        "teacher_output_sha256": [first.raw_sha256, second.raw_sha256],
        "prompt_sha256": first.prompt_sha256,
    }
    if first.prompt_sha256 != second.prompt_sha256:
        provenance["prompt_sha256"] = [first.prompt_sha256, second.prompt_sha256]
    return {
        "admitted": True,
        "reason": "two_teacher_consensus",
        "target": target,
        "reason_code": first.reason_code,
        "same_product_type": bool(first.same_product_type and second.same_product_type),
        **provenance,
    }


__all__ = ["VERDICTS", "TeacherDecision", "consensus_label"]
