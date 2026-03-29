from dataclasses import dataclass

@dataclass
class ClinicalCaseDTO:
    """The 'Non-Vector' version of a patient case for the Logic Engine."""
    case_id: int
    name: str
    specialty: str
    priority_score: float
    is_critical: bool
    requires_immediate_transport: bool
    is_unstable: bool

class LogicBridge:
    def __init__(self, priority_manager):
        # Your PatientPriorityManager contains the queue and scaler
        self.pm = priority_manager

    def prepare_for_logic_engine(self, patient_obj) -> ClinicalCaseDTO:
        """
        Translates a Patient object (Vector-based) into a 
        ClinicalCaseDTO (Semantic-based).
        """
        # 1. Access the vector indices from the Patient object
        # index 0: urgency, index 2: stability, index 3: distance
        urgency = patient_obj.vector[0]
        stability = patient_obj.vector[2]
        distance = patient_obj.vector[3]
        
        # 2. Translate vector dimensions into Boolean/Medical logic
        # We use the scaled 0-10 values here
        is_critical = patient_obj.priority_score >= 0.85
        is_unstable = stability < 3.0  # Low stability value = high clinical risk
        needs_transport = distance > 7.0 # Represents long physical distance
        
        return ClinicalCaseDTO(
            case_id=patient_obj.case_id,
            name=patient_obj.name,
            specialty=patient_obj.specialty,
            priority_score=patient_obj.priority_score,
            is_critical=is_critical,
            requires_immediate_transport=needs_transport,
            is_unstable=is_unstable
        )
