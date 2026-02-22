from dataclasses import dataclass

@dataclass
class ClinicalCaseDTO:
    """The 'Non-Vector' version of a patient case for the Logic Engine."""
    case_id: int
    specialty: str
    is_critical: bool
    requires_immediate_transport: bool
    # Add other 'human-readable' fields here

class LogicBridge:
    def __init__(self, vector_manager):
        self.vm = vector_manager

    def prepare_for_logic_engine(self, case_id: int) -> ClinicalCaseDTO:
        # 1. Fetch the vector object from your existing manager
        vector_obj = self.vm.get_case_vector(case_id)
        
        # 2. Translate vector logic into Boolean/Medical logic
        # This is where you 'rewrite' the data to be non-vector based
        is_critical = vector_obj.urgency_score >= 8
        needs_transport = vector_obj.distance_km > 50
        
        return ClinicalCaseDTO(
            case_id=vector_obj.case_id,
            specialty=vector_obj.specialty_name(),
            is_critical=is_critical,
            requires_immediate_transport=needs_transport
        )
