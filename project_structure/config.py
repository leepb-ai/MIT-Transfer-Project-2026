# healthcomm/config.py
#
# Leaf-node config: zero imports of its own.
# Every other module in the healthcomm package imports from here.
# Change a threshold once — all consumers see it.

from dataclasses import dataclass


@dataclass(frozen=True)
class ClinicalConfig:
    """
    Immutable runtime configuration for the healthcomm pipeline.

    frozen=True ensures no threshold can be mutated after construction,
    which is critical in a clinical system where a mid-execution change
    could silently alter routing decisions.

    Two confidence thresholds are intentional and distinct:
      BRIDGE_MIN_CONFIDENCE  — bridge layer: "is this vector match weak enough to flag?"
      ROUTER_MIN_CONFIDENCE  — router layer: "is this case confident enough to act on automatically?"
    The router applies a stricter bar because flagging and acting carry different risk levels.
    """

    # ── Bridge layer ──────────────────────────────────────────────────────────
    BRIDGE_MIN_CONFIDENCE: float = 0.3   # Below this → low_confidence flag set on DTO.
                                          # Case still passes through — never discarded.
    URGENCY_THRESHOLD: float = 8.0        # urgency_score >= this → is_critical = True
    TRANSPORT_KM_THRESHOLD: float = 50.0  # distance_km > this → requires_immediate_transport = True

    # ── Router layer ──────────────────────────────────────────────────────────
    ROUTER_MIN_CONFIDENCE: float = 0.5   # Below this → GRAY regardless of other signals.
                                          # A case can be low_confidence=False at the bridge
                                          # and still hit GRAY here — two distinct bars.
