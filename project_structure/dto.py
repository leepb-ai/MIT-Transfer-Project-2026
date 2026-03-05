# healthcomm/models/dto.py
#
# Typed contract that crosses every layer boundary in the healthcomm pipeline.
# No business logic lives here beyond predicates that describe the DTO's own state.

from dataclasses import dataclass


@dataclass
class ClinicalCaseDTO:
    # ── Core identity ─────────────────────────────────────────────────────────
    case_id: int
    specialty: str                   # Canonical Clinical DB value or GENERAL_PRACTICE
    manual_review_required: bool     # Required field — caller must set explicitly.
                                     # True overrides all routing; case goes to GRAY
                                     # regardless of confidence or criticality.

    # ── Clinical signals ──────────────────────────────────────────────────────
    is_critical: bool
    requires_immediate_transport: bool
    evidence_text: str               # Normalised snippet; "No evidence available" sentinel
                                     # when source is empty. Triggers a user notification
                                     # in the router but does NOT change the route.

    # ── Data-quality signals ──────────────────────────────────────────────────
    confidence_score: float = 0.0
    low_confidence: bool = False     # True when similarity < BRIDGE_MIN_CONFIDENCE.
                                     # Does not gate routing alone; attaches a warning
                                     # and maps to requires_human_review().
    unknown_specialty: bool = False  # True when raw specialty could not be mapped.
                                     # Pushes route to GRAY so a human can verify
                                     # before the case is acted on.

    # ── Predicates ────────────────────────────────────────────────────────────

    def is_actionable(self) -> bool:
        """True when the case warrants any routing action at all.

        Includes flagged cases — consumers must check quality flags
        and apply appropriate caution rather than discarding.
        """
        return self.is_critical or self.requires_immediate_transport

    def requires_human_review(self) -> bool:
        """True when a case is actionable but carries any data-quality flag.

        low_confidence alone attaches a warning (route is unchanged).
        unknown_specialty pushes to GRAY.
        manual_review_required unconditionally pushes to GRAY.

        This predicate is the single place that encodes "human needed" logic
        so audit tools and the Logic Engine can query it without knowing
        which specific flags are set.
        """
        return self.is_actionable() and (
            self.low_confidence
            or self.unknown_specialty
            or self.manual_review_required
        )

    def has_evidence(self) -> bool:
        """False when evidence_text carries the sentinel value.

        Used by the router to trigger a no-evidence notification without
        coupling the routing decision to evidence quality.
        """
        return self.evidence_text != "No evidence available"
