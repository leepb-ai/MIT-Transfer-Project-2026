#!/usr/bin/env python3

"""
HealthComm Hybrid Priority System
---------------------------------
A vector-based triage engine designed for low-connectivity clinical environments.
This system uses 4D patient vectors to calculate priority scores based on
specialty reference anchors and real-time patient data.

Dimensions: [Urgency, Time, Stability, Distance]
"""

import heapq
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Database dependency check for portability
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

# Reference Vectors: Idealized "Critical" states for different medical specialties
# Used as anchors for Cosine Similarity calculations.
SPECIALTY_REFERENCES = {
    "Cardiology": np.array([10.0, 3.0, 10.0, 2.0]),
    "Pediatrics": np.array([10.0, 5.0, 7.0, 8.0]),
    "ER":         np.array([10.0, 5.0, 8.0, 3.0]),
    "Dental":     np.array([8.0, 2.0, 3.0, 5.0])
}

@dataclass
class Patient:
    """
    Represents a clinical case within the triage system.
    Calculates a priority score using a hybrid of Cosine Similarity and Magnitude.
    """
    patient_id: int
    name: str
    specialty: str
    vector: np.ndarray  # 4D vector: [Urgency, Time, Stability, Distance]
    case_id: Optional[int] = None
    chief_complaint: Optional[str] = None
    
    priority_score: float = field(init=False)
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        # Validation for 4D Vector Space Requirement
        if self.vector.shape[0] != 4:
            raise ValueError(f"Vector must be 4D, got {self.vector.shape[0]}D")
        
        if self.specialty not in SPECIALTY_REFERENCES:
            raise ValueError(f"Invalid specialty: {self.specialty}")
        
        self.priority_score = self.calculate_priority()
    
    def calculate_priority(self) -> float:
        """
        The Core Triage Algorithm:
        Combines Cosine Similarity (Direction) and Vector Magnitude (Magnitude).
        Score = (0.6 * Similarity) + (0.4 * Severity Ratio)
        """
        reference = SPECIALTY_REFERENCES[self.specialty]
        
        # Normalize vectors for Cosine Similarity
        reference_magnitude = np.linalg.norm(reference)
        unit_reference = reference / reference_magnitude if reference_magnitude != 0 else reference
        
        patient_magnitude = np.linalg.norm(self.vector)
        unit_patient = self.vector / patient_magnitude if patient_magnitude != 0 else self.vector
        
        # Calculate Directional Alignment (0.0 to 1.0)
        cosine_similarity = np.dot(unit_reference, unit_patient)
        
        # Calculate Magnitude Ratio (How 'large' is the patient's distress vs reference)
        magnitude_ratio = patient_magnitude / reference_magnitude if reference_magnitude != 0 else 1.0
        
        # Weighted Final Priority Score
        priority_score = (0.6 * cosine_similarity) + (0.4 * magnitude_ratio)
        
        return float(priority_score)
    
    def detect_single_spike(self) -> Tuple[bool, int, float]:
        """
        Heuristic to detect if a single dimension has reached a critical threshold (>7.0),
        even if the overall priority score remains moderate.
        """
        threshold = 7.0
        high_dims = [(i, v) for i, v in enumerate(self.vector) if v > threshold]
        
        if len(high_dims) == 1:
            dim_index, dim_value = high_dims[0]
            return True, dim_index, dim_value
        
        return False, -1, 0.0
    
    def get_dimension_name(self, index: int) -> str:
        """Maps vector indices to human-readable clinical labels."""
        names = ['urgency', 'time', 'stability', 'distance']
        return names[index] if 0 <= index < 4 else 'unknown'
    
    def __lt__(self, other):
        """Used by the heap queue to maintain priority order (Max-Heap simulation)."""
        if self.priority_score != other.priority_score:
            return self.priority_score > other.priority_score
        
        # Tie-breaker: FIFO (First In, First Out)
        return self.timestamp < other.timestamp

class AlertThrottler:
    """
    Prevents 'Alert Fatigue' by limiting the frequency of critical notifications.
    Standard limit is 1 alert per 0.5 seconds.
    """
    def __init__(self, limit: float = 0.5):
        self.limit = limit
        self.last_alert_time = 0.0
    
    def can_alert(self, current_time: Optional[float] = None) -> bool:
        if current_time is None:
            current_time = time.time()
        
        if current_time - self.last_alert_time >= self.limit:
            self.last_alert_time = current_time
            return True
        
        return False

class BufferedPriorityQueue:
    """
    A priority-aware data structure for managing patient flow.
    Wraps Python's heapq to provide clinical alerting and performance metrics.
    """
    def __init__(self, alert_throttler: Optional[AlertThrottler] = None):
        self.heap: List[Patient] = []
        self.throttler = alert_throttler or AlertThrottler(limit=0.5)
        self.processed_count = 0
        self.total_wait_time = 0.0
    
    def add_patient(self, patient: Patient) -> bool:
        """Adds patient to queue and triggers alerts for ultra-high priority cases."""
        heapq.heappush(self.heap, patient)
        
        # Trigger critical alert if priority exceeds 90%
        if patient.priority_score >= 0.9 and self.throttler.can_alert():
            self._trigger_alert(patient)
        
        return True
    
    def get_next_patient(self) -> Optional[Patient]:
        """Removes and returns the highest priority patient."""
        if self.heap:
            patient = heapq.heappop(self.heap)
            
            # Metrics tracking
            wait_time = time.time() - patient.timestamp
            self.total_wait_time += wait_time
            self.processed_count += 1
            
            return patient
        return None

class VectorScaler:
    """
    Normalizes raw clinical data into the standard [0, 10] triage scale.
    Ensures that different units (hours, km, scores) are comparable in vector space.
    """
    def __init__(self):
        self.bounds = {
            'urgency':   (1, 10),
            'time':      (0, 24),    # Max 24h wait
            'stability': (-10, 10),  # Scale from declining to improving
            'distance':  (0, 100)    # Distance in km
        }
    
    def scale_to_0_10(self, value: float, feature_name: str) -> float:
        """Min-Max Scaling implementation."""
        f_min, f_max = self.bounds.get(feature_name)
        clamped = max(f_min, min(value, f_max))
        normalized = (clamped - f_min) / (f_max - f_min)
        return float(normalized * 10.0)

    def create_patient_vector(self, urgency, time_hours, stability, distance_km) -> np.ndarray:
        """Generates the final 4D vector for the triage engine."""
        return np.array([
            self.scale_to_0_10(urgency, 'urgency'),
            self.scale_to_0_10(time_hours, 'time'),
            self.scale_to_0_10(stability, 'stability'),
            self.scale_to_0_10(distance_km, 'distance')
        ])

class PatientPriorityManager:
    """
    The Orchestrator class. Links the Database (PostgreSQL), 
    the Vector Scaling logic, and the Priority Queue.
    """
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.scaler = VectorScaler()
        self.queue = BufferedPriorityQueue()
    
    # ... (Database methods remain consistent with your logic bridge design)
