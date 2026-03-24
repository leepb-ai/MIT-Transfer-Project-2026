#!/usr/bin/env python3
"""
HealthComm Patient Case Vector Manager(Vector embedding Paradigm)
=======================================
Handles patient cases represented as 5-dimensional vectors.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SpecialtyEmbedding:
    """
    Replaces raw integer codes with semantic vectors.
    Dimensions: [Resource Intensity, Criticality Bias, Equipment Requirement]
    """
    name: str
    vector: np.ndarray 

class HealthCommVectorCore:
    # Defining our semantic space for specialties
    # Cardiology and ER are closer in 'Criticality Bias'
    SPECIALTIES = {
        1: SpecialtyEmbedding("Cardiology", np.array([0.9, 0.9, 0.8])),
        2: SpecialtyEmbedding("Pediatrics", np.array([0.4, 0.3, 0.2])),
        3: SpecialtyEmbedding("ER",         np.array([0.8, 1.0, 0.9])),
        4: SpecialtyEmbedding("Dental",     np.array([0.3, 0.1, 0.6]))
    }

    @staticmethod
    def get_embedding(code: int) -> np.ndarray:
        return HealthCommVectorCore.SPECIALTIES.get(code).vector

@dataclass
class PatientCaseVector:
    urgency_score: int         # x1
    specialty_code: int        # x2 (Used to fetch embedding)
    time_intensity: float      # x3
    stability_trend: float     # x4
    distance_km: float         # x5
    
    def to_numpy(self) -> np.ndarray:
        """
        Generates the dense vector. 
        We replace the single x2 with its 3D embedding.
        Resulting Vector: [x1, e1, e2, e3, x3, x4, x5] (7D Space)
        """
        spec_vector = HealthCommVectorCore.get_embedding(self.specialty_code)
        
        # Concatenate scalar metrics with the specialty embedding
        return np.concatenate([
            [self.urgency_score],
            spec_vector,
            [self.time_intensity, self.stability_trend, self.distance_km]
        ])

    def cosine_similarity(self, other: 'PatientCaseVector') -> float:
        """
        Measures the cosine of the angle between two case vectors.
        1.0 = Identical direction (similar medical profile)
        0.0 = Orthogonal (completely unrelated cases)
        """
        v1 = self.to_numpy()
        v2 = other.to_numpy()
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))






