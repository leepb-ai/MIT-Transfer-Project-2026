#!/usr/bin/env python3
"""
HealthComm Patient Case Vector Manager
=======================================
Handles patient cases represented as 5-dimensional vectors.

Vector: Patient_case = (x1, x2, x3, x4, x5)
  x1 = Urgency Score (1-10)
  x2 = Specialty Code (1=Cardiology, 2=Pediatrics, 3=ER, 4=Dental)
  x3 = Intensity of Time (hours)
  x4 = Stability Trend (negative=improving, positive=worsening)
  x5 = Distance (km from provider)
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PatientCaseVector:
    """Represents a patient case as a 5-dimensional vector."""
    
    urgency_score: int        # x1: 1-10
    specialty_code: int       # x2: 1-4
    time_intensity: float     # x3: hours
    stability_trend: float    # x4: rate of change
    distance_km: float        # x5: kilometers
    
    case_id: Optional[int] = None
    patient_id: Optional[int] = None
    chief_complaint: Optional[str] = None
    priority_level: Optional[str] = None
    
    def __post_init__(self):
        """Validate vector components."""
        if not 1 <= self.urgency_score <= 10:
            raise ValueError("Urgency score must be between 1 and 10")
        if not 1 <= self.specialty_code <= 4:
            raise ValueError("Specialty code must be between 1 and 4")
        if self.time_intensity < 0:
            raise ValueError("Time intensity must be non-negative")
        if self.distance_km < 0:
            raise ValueError("Distance must be non-negative")
    
    def to_numpy(self) -> np.ndarray:
        """Convert vector to numpy array."""
        return np.array([
            self.urgency_score,
            self.specialty_code,
            self.time_intensity,
            self.stability_trend,
            self.distance_km
        ])
    
    def magnitude(self) -> float:
        """Calculate Euclidean norm (magnitude) of the vector."""
        return np.linalg.norm(self.to_numpy())
    
    def __repr__(self) -> str:
        return (f"PatientCaseVector(urgency={self.urgency_score}, "
                f"specialty={self.specialty_code}, "
                f"time={self.time_intensity:.1f}h, "
                f"trend={self.stability_trend:+.1f}, "
                f"dist={self.distance_km:.1f}km, "
                f"||v||={self.magnitude():.2f})")


class PatientCaseVectorManager:
    """Manages patient case vectors and vector operations."""
    
    SPECIALTY_NAMES = {
        1: "Cardiology",
        2: "Pediatrics",
        3: "Emergency Room",
        4: "Dental Care"
    }
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
    
    def get_connection(self):
        return psycopg2.connect(**self.db_config)
    
    def create_case(self, patient_id: int, created_by_user_id: int,
                   urgency_score: int, specialty_code: int,
                   time_intensity: float, stability_trend: float,
                   distance_km: float, chief_complaint: str,
                   case_description: str = "") -> Optional[int]:
        """Create a new patient case with vector representation."""
        vector = PatientCaseVector(
            urgency_score=urgency_score,
            specialty_code=specialty_code,
            time_intensity=time_intensity,
            stability_trend=stability_trend,
            distance_km=distance_km
        )
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO patient_cases (
                        patient_id, created_by_user_id,
                        urgency_score, specialty_code, time_intensity,
                        stability_trend, distance_km,
                        chief_complaint, case_description
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING case_id, priority_level, vector_magnitude
                    """,
                    (patient_id, created_by_user_id,
                     urgency_score, specialty_code, time_intensity,
                     stability_trend, distance_km,
                     chief_complaint, case_description)
                )
                result = cursor.fetchone()
                case_id, priority, magnitude = result
                conn.commit()
                
                print(f"✓ Case created: ID={case_id}, Priority={priority}, ||v||={magnitude:.2f}")
                return case_id
        except Exception as e:
            conn.rollback()
            print(f"✗ Error: {e}")
            return None
        finally:
            conn.close()
    
    def get_case_vector(self, case_id: int) -> Optional[PatientCaseVector]:
        """Retrieve a case as a vector object."""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT case_id, patient_id, urgency_score, specialty_code,
                           time_intensity, stability_trend, distance_km,
                           chief_complaint, priority_level
                    FROM patient_cases WHERE case_id = %s
                    """, (case_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return None
                
                return PatientCaseVector(
                    case_id=row['case_id'],
                    patient_id=row['patient_id'],
                    urgency_score=row['urgency_score'],
                    specialty_code=row['specialty_code'],
                    time_intensity=float(row['time_intensity']),
                    stability_trend=float(row['stability_trend']),
                    distance_km=float(row['distance_km']),
                    chief_complaint=row['chief_complaint'],
                    priority_level=row['priority_level']
                )
        finally:
            conn.close()
    
    def euclidean_distance(self, case_id_a: int, case_id_b: int) -> Optional[float]:
        """Calculate Euclidean distance between two case vectors."""
        vector_a = self.get_case_vector(case_id_a)
        vector_b = self.get_case_vector(case_id_b)
        
        if not vector_a or not vector_b:
            return None
        
        return np.linalg.norm(vector_a.to_numpy() - vector_b.to_numpy())
    
    def get_critical_cases(self) -> List[Dict]:
        """Get all open critical cases."""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT case_id, patient_id, chief_complaint,
                           urgency_score, specialty_code, time_intensity,
                           stability_trend, distance_km, vector_magnitude,
                           priority_level
                    FROM patient_cases
                    WHERE priority_level = 'CRITICAL' AND current_status = 'Open'
                    ORDER BY urgency_score DESC, vector_magnitude DESC
                    """
                )
                return cursor.fetchall()
        finally:
            conn.close()


def main():
    """Demo of patient case vector operations."""
    
    db_config = {
        'host': 'localhost',
        'database': 'healthcomm',
        'user': 'postgres',
        'password': 'your_password',
        'port': 5432
    }
    
    mgr = PatientCaseVectorManager(db_config)
    
    print("\n" + "="*70)
    print("PATIENT CASE VECTOR DEMO")
    print("="*70)
    
    # Create a critical cardiac case
    print("\n1. Creating critical cardiac case...")
    case_id = mgr.create_case(
        patient_id=1006,
        created_by_user_id=4,
        urgency_score=10,      # Maximum urgency
        specialty_code=1,       # Cardiology
        time_intensity=6.0,     # 6 hours
        stability_trend=3.5,    # Rapidly worsening
        distance_km=2.5,        # 2.5 km
        chief_complaint="Acute MI with complications"
    )
    
    # Get as vector
    print(f"\n2. Vector representation:")
    vector = mgr.get_case_vector(case_id)
    print(f"   {vector}")
    
    # Get all critical cases
    print("\n3. All CRITICAL cases:")
    critical = mgr.get_critical_cases()
    for case in critical:
        print(f"   Case {case['case_id']}: {case['chief_complaint']}")
        print(f"   Vector: ({case['urgency_score']}, {case['specialty_code']}, "
              f"{case['time_intensity']}, {case['stability_trend']}, {case['distance_km']})")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
