# healthcomm/action_router.py

from enum import Enum, auto
import logging

from healthcomm.config import ClinicalConfig
from healthcomm.models.dto import ClinicalCaseDTO

logger = logging.getLogger(__name__)


class RoutePriority(Enum):
    RED    = auto()   # Critical or requires immediate transport
    YELLOW = auto()   # Specialist case, not critical, sufficient confidence
    GREEN  = auto()   # General practice, not critical, sufficient confidence
    GRAY   = auto()   # Insufficient confidence, unknown specialty, or manual flag
                      # → human review required before any automated action


class ActionRouter:
    """
    Logic Engine layer: translates a ClinicalCaseDTO into a RoutePriority
    and dispatches the appropriate protocol.

    determine_route() and execute_route() are intentionally separate:
    callers that only need the priority (e.g. for display) can call
    determine_route() without triggering any side effects.

    Confidence thresholds:
      ROUTER_MIN_CONFIDENCE (0.5) is stricter than BRIDGE_MIN_CONFIDENCE (0.3).
      A case can pass the bridge (low_confidence=False) and still hit GRAY here.
      See ClinicalConfig for the rationale.
    """

    def __init__(self, clinical_db, config: ClinicalConfig, notifier=None):
        """
        Args:
            clinical_db:  Clinical DB client (Hour 3 integration).
            config:       Shared ClinicalConfig instance (thresholds).
            notifier:     Optional callable(case_id, message) for user-facing
                          notifications (e.g. no-evidence warning).
        """
        self.db = clinical_db
        self.cfg = config
        self.notify = notifier

    # ── Public API ────────────────────────────────────────────────────────────

    def determine_route(self, dto: ClinicalCaseDTO) -> RoutePriority:
        """
        Routing rules (evaluated in priority order):

        GRAY  — safety gate, checked first:
                  • manual_review_required flag is set, OR
                  • confidence_score < ROUTER_MIN_CONFIDENCE, OR
                  • unknown_specialty flag is set
                Any of these means a human must verify before automated action.

        RED   — critical cases:
                  • is_critical OR requires_immediate_transport
                  • low_confidence here attaches a warning but does NOT downgrade
                    to GRAY (confidence passed the router gate above).

        YELLOW — specialist, non-critical:
                  • specialty != GENERAL_PRACTICE
                  • low_confidence attaches a warning; maps to requires_human_review()
                    on the DTO so audit tools can track it.

        GREEN  — default safe path:
                  • general practice, not critical, sufficient confidence.
        """
        # 1. SAFETY GATE — human must review before any automated action
        if (
            dto.manual_review_required
            or dto.confidence_score < self.cfg.ROUTER_MIN_CONFIDENCE
            or dto.unknown_specialty
        ):
            return RoutePriority.GRAY

        # Issue notifications for quality flags that don't change the route
        self._check_quality_warnings(dto)

        # 2. EMERGENCY PATH
        if dto.is_critical or dto.requires_immediate_transport:
            return RoutePriority.RED

        # 3. SPECIALIST PATH — non-critical but not general practice
        if dto.specialty != "GENERAL_PRACTICE":
            return RoutePriority.YELLOW

        # 4. DEFAULT SAFE PATH
        return RoutePriority.GREEN

    def execute_route(self, priority: RoutePriority, dto: ClinicalCaseDTO) -> None:
        """Dispatches the case to the correct system or DB table."""
        if priority == RoutePriority.RED:
            self._trigger_emergency_protocol(dto)
        elif priority == RoutePriority.YELLOW:
            self._notify_specialist(dto)
        elif priority == RoutePriority.GREEN:
            self._schedule_standard_review(dto)
        elif priority == RoutePriority.GRAY:
            self._flag_for_human_review(dto)

    # ── Private dispatch stubs ────────────────────────────────────────────────
    # Replace with real Clinical DB calls in Hour 3.

    def _trigger_emergency_protocol(self, dto: ClinicalCaseDTO) -> None:
        logger.critical("RED — Emergency protocol triggered for Case %s.", dto.case_id)

    def _notify_specialist(self, dto: ClinicalCaseDTO) -> None:
        logger.info("YELLOW — Specialist notification queued for Case %s (%s).",
                    dto.case_id, dto.specialty)

    def _schedule_standard_review(self, dto: ClinicalCaseDTO) -> None:
        logger.info("GREEN — Standard review scheduled for Case %s.", dto.case_id)

    def _flag_for_human_review(self, dto: ClinicalCaseDTO) -> None:
        logger.warning("GRAY — Case %s held for human review.", dto.case_id)

    # ── Quality warning helpers ───────────────────────────────────────────────

    def _check_quality_warnings(self, dto: ClinicalCaseDTO) -> None:
        """Issue notifications for data-quality issues that don't change the route.

        Called after the GRAY gate — these warnings apply only to cases
        that passed the confidence and specialty checks.
        """
        if not dto.has_evidence():
            msg = (
                f"Case {dto.case_id}: no clinical evidence available — "
                f"case will proceed on clinical signals alone. Manual verification recommended."
            )
            logger.warning(msg)
            if self.notify:
                self.notify(dto.case_id, msg)

        if dto.low_confidence:
            msg = (
                f"Case {dto.case_id}: confidence score ({dto.confidence_score:.2f}) "
                f"passed the routing gate but is below the bridge threshold — "
                f"treat routing outcome with caution."
            )
            logger.warning(msg)
            if self.notify:
                self.notify(dto.case_id, msg)
