# healthcomm/logic_bridge.py

from typing import Optional
import logging

from healthcomm.config import ClinicalConfig
from healthcomm.models.dto import ClinicalCaseDTO

logger = logging.getLogger(__name__)

# Maximum characters preserved from a vector's text_content field.
# Prevents arbitrarily large strings from crossing the bridge layer boundary.
EVIDENCE_TEXT_MAX_LEN = 500


class LogicBridge:
    """
    Bridge layer between the VectorManager and the Logic Engine.

    Responsibilities:
    - Safely fetch vector objects from the VectorManager
    - Normalise raw specialty strings to canonical Clinical DB values
    - Map raw vector data to well-typed ClinicalCaseDTOs
    - Apply threshold logic for criticality and transport
    - Flag (but never discard) low-confidence or unmappable-specialty cases
    - Notify consumers of data-quality issues via structured log events

    Explicitly NOT responsible for:
    - Clinical routing decisions  (Logic Engine / ActionRouter)
    - Specialty-specific threshold overrides  (Logic Engine / config layer)
    - Transport dispatch  (downstream transport services)
    """

    # Expand as the Clinical DB grows.
    # Keys must be UPPERCASE. Values must match canonical Clinical DB specialty IDs.
    # Externalise to a config file or DB table once entries exceed ~20.
    SPECIALTY_MAP: dict = {
        "CARDIO":    "CARDIOLOGY",
        "HEART":     "CARDIOLOGY",
        "CARDIAC":   "CARDIOLOGY",
        "NEURO":     "NEUROLOGY",
        "NEUROLOGY": "NEUROLOGY",
        "ORTHO":     "ORTHOPAEDICS",
        "GENERAL":   "GENERAL_PRACTICE",
        "GP":        "GENERAL_PRACTICE",
    }

    FALLBACK_SPECIALTY = "GENERAL_PRACTICE"

    def __init__(self, vector_manager, config: ClinicalConfig, error_notifier=None):
        """
        Args:
            vector_manager:  Provides get_case_vector(case_id).
            config:          Shared ClinicalConfig instance (thresholds).
            error_notifier:  Optional callable(case_id, message) for surfacing
                             data-quality issues to the user/UI layer.
                             If None, issues are logged only.
        """
        self.vm = vector_manager
        self.cfg = config
        self.notify = error_notifier

    def prepare_for_logic_engine(self, case_id: int) -> Optional[ClinicalCaseDTO]:
        """Main entry point.

        Returns a ClinicalCaseDTO or None if the vector cannot be found.
        Low-confidence and unknown-specialty cases are passed through with
        their respective flags set rather than being dropped.
        """
        vector_obj = self._fetch_vector(case_id)
        if vector_obj is None:
            return None

        dto = self._map_to_dto(vector_obj)
        self._audit_flags(dto)
        return dto

    # -- Private helpers -------------------------------------------------------

    def _fetch_vector(self, case_id: int):
        vector_obj = self.vm.get_case_vector(case_id)
        if not vector_obj:
            logger.error("Vector for Case %s not found — audit trail flagged.", case_id)
        return vector_obj

    def _normalize_specialty(self, raw_name: str) -> tuple:
        """Returns (canonical_specialty, unknown_specialty_flag).

        Strips, uppercases, and looks up raw_name in SPECIALTY_MAP.
        Falls back to FALLBACK_SPECIALTY and sets the flag when unmapped.
        """
        clean = raw_name.upper().strip() if raw_name else ""
        if clean in self.SPECIALTY_MAP:
            return self.SPECIALTY_MAP[clean], False
        return self.FALLBACK_SPECIALTY, True

    @staticmethod
    def _normalize_evidence(raw_text: str) -> str:
        """Strips whitespace and truncates to EVIDENCE_TEXT_MAX_LEN characters.

        Returns the sentinel string 'No evidence available' for empty/None input.
        The router checks has_evidence() on the DTO and issues a user notification
        when the sentinel is present — the route itself is unchanged.
        """
        if not raw_text:
            return "No evidence available"
        cleaned = raw_text.strip()
        if len(cleaned) > EVIDENCE_TEXT_MAX_LEN:
            return cleaned[:EVIDENCE_TEXT_MAX_LEN] + "..."
        return cleaned

    def _map_to_dto(self, vector_obj) -> ClinicalCaseDTO:
        confidence = getattr(vector_obj, "similarity", 0.0)
        distance   = getattr(vector_obj, "distance_km", 0) or 0  # guard None & missing
        specialty, unknown = self._normalize_specialty(vector_obj.specialty_name())

        return ClinicalCaseDTO(
            case_id=vector_obj.case_id,
            specialty=specialty,
            manual_review_required=getattr(vector_obj, "manual_review_required", False),
            is_critical=vector_obj.urgency_score >= self.cfg.URGENCY_THRESHOLD,
            requires_immediate_transport=distance > self.cfg.TRANSPORT_KM_THRESHOLD,
            evidence_text=self._normalize_evidence(
                getattr(vector_obj, "text_content", "")
            ),
            confidence_score=confidence,
            low_confidence=confidence < self.cfg.BRIDGE_MIN_CONFIDENCE,
            unknown_specialty=unknown,
        )

    def _audit_flags(self, dto: ClinicalCaseDTO) -> None:
        """Log and optionally notify the user of any data-quality flags on the DTO."""
        if dto.manual_review_required:
            msg = (
                f"Case {dto.case_id}: manual_review_required flag is set — "
                f"case will be routed to GRAY regardless of confidence or criticality."
            )
            logger.warning(msg)
            if self.notify:
                self.notify(dto.case_id, msg)

        if dto.low_confidence:
            msg = (
                f"Case {dto.case_id}: low confidence score ({dto.confidence_score:.2f}) "
                f"— case flagged for human review, not discarded."
            )
            logger.warning(msg)
            if self.notify:
                self.notify(dto.case_id, msg)

        if dto.unknown_specialty:
            msg = (
                f"Case {dto.case_id}: specialty could not be mapped "
                f"— defaulted to {self.FALLBACK_SPECIALTY}. "
                f"Add the raw value to SPECIALTY_MAP to resolve."
            )
            logger.error(msg)
            if self.notify:
                self.notify(dto.case_id, msg)
