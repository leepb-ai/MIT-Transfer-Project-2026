#!/usr/bin/env python3

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


class VectorScaler:

    def __init__(self):
        self.bounds = {
            'urgency':   (1, 10),
            'hours':     (0, 24),
            'stability': (-10, 10),
            'distance':  (0, 100)
        }

    def scale_feature(self, value, feature_name):
        f_min, f_max = self.bounds.get(feature_name)
        clamped_val = max(f_min, min(value, f_max))
        return (clamped_val - f_min) / (f_max - f_min)

    def transform_patient_vector(self, raw_data: dict) -> np.ndarray:
        scaled_urgency   = self.scale_feature(raw_data['urgency_score'],   'urgency')
        scaled_time      = self.scale_feature(raw_data['time_intensity'],  'hours')
        scaled_stability = self.scale_feature(raw_data['stability_trend'], 'stability')
        scaled_distance  = self.scale_feature(raw_data['distance_km'],     'distance')

        one_hot = one_hot_encode_specialty(raw_data['specialty_code'])

        vector = np.array([
            scaled_urgency,
            float(one_hot[0]),
            float(one_hot[1]),
            float(one_hot[2]),
            float(one_hot[3]),
            scaled_time,
            scaled_stability,
            scaled_distance
        ])

        return vector


SPECIALTY_MAP = {
    1: "Cardiology",
    2: "Pediatrics",
    3: "ER",
    4: "Dental"
}

NUM_SPECIALTIES = len(SPECIALTY_MAP)


def one_hot_encode_specialty(specialty_code: int) -> np.ndarray:
    if specialty_code not in SPECIALTY_MAP:
        raise ValueError(
            f"Invalid specialty code: {specialty_code}. "
            f"Must be one of {list(SPECIALTY_MAP.keys())}"
        )
    one_hot = np.zeros(NUM_SPECIALTIES, dtype=int)
    one_hot[specialty_code - 1] = 1
    return one_hot


def decode_one_hot_specialty(one_hot: np.ndarray) -> int:
    index = np.argmax(one_hot)
    return int(index + 1)


@dataclass
class PatientCaseVector:

    urgency_score:   int
    specialty_code:  int
    time_intensity:  float
    stability_trend: float
    distance_km:     float

    case_id:         Optional[int] = None
    patient_id:      Optional[int] = None
    chief_complaint: Optional[str] = None
    priority_level:  Optional[str] = None

    def __post_init__(self):
        if not 1 <= self.urgency_score <= 10:
            raise ValueError(f"Urgency score must be 1-10, got: {self.urgency_score}")
        if self.specialty_code not in SPECIALTY_MAP:
            raise ValueError(f"Specialty code must be 1-4, got: {self.specialty_code}")
        if self.time_intensity < 0:
            raise ValueError(f"Time intensity must be >= 0, got: {self.time_intensity}")
        if self.distance_km < 0:
            raise ValueError(f"Distance must be >= 0, got: {self.distance_km}")

    def get_one_hot_specialty(self) -> np.ndarray:
        return one_hot_encode_specialty(self.specialty_code)

    def to_numpy(self) -> np.ndarray:
        one_hot = self.get_one_hot_specialty()
        vector = np.array([
            float(self.urgency_score),
            float(one_hot[0]),
            float(one_hot[1]),
            float(one_hot[2]),
            float(one_hot[3]),
            float(self.time_intensity),
            float(self.stability_trend),
            float(self.distance_km)
        ])
        return vector

    def magnitude(self) -> float:
        return float(np.linalg.norm(self.to_numpy()))

    def specialty_name(self) -> str:
        return SPECIALTY_MAP[self.specialty_code]

    def __repr__(self) -> str:
        one_hot = self.get_one_hot_specialty()
        one_hot_str = f"[{','.join(str(int(x)) for x in one_hot)}]"
        return (
            f"PatientCaseVector(\n"
            f"  x1 (urgency)    = {self.urgency_score}/10\n"
            f"  x2 (specialty)  = {one_hot_str} → {self.specialty_name()}\n"
            f"  x3 (time)       = {self.time_intensity:.1f} hours\n"
            f"  x4 (trend)      = {self.stability_trend:+.2f}\n"
            f"  x5 (distance)   = {self.distance_km:.1f} km\n"
            f"  full vector     = {self.to_numpy()}\n"
            f"  ||v||           = {self.magnitude():.4f}\n"
            f"  priority        = {self.priority_level or 'Not computed'}\n"
            f")"
        )


class PatientCaseVectorManager:

    VECTOR_DIMENSION = 8

    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config

    def get_connection(self):
        return psycopg2.connect(**self.db_config)

    def create_case(self,
                    patient_id: int,
                    created_by_user_id: int,
                    urgency_score: int,
                    specialty_code: int,
                    time_intensity: float,
                    stability_trend: float,
                    distance_km: float,
                    chief_complaint: str,
                    case_description: str = ""
                    ) -> Optional[int]:

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
                     urgency_score, specialty_code,
                     time_intensity, stability_trend,
                     distance_km, chief_complaint, case_description)
                )

                result = cursor.fetchone()
                case_id, priority, magnitude = result
                conn.commit()

                print(f"✓ Case {case_id} created!")
                print(f"  One-Hot Vector: {vector.to_numpy()}")
                print(f"  Specialty: {vector.specialty_name()} "
                      f"→ One-Hot: {vector.get_one_hot_specialty()}")
                print(f"  Priority: {priority}, ||v||={float(magnitude):.4f}")

                return case_id

        except ValueError as e:
            print(f"✗ Validation error: {e}")
            return None

        except psycopg2.Error as e:
            conn.rollback()
            print(f"✗ Database error: {e}")
            return None

        finally:
            conn.close()

    def get_case_vector(self, case_id: int) -> Optional[PatientCaseVector]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT case_id, patient_id,
                           urgency_score, specialty_code,
                           time_intensity, stability_trend, distance_km,
                           chief_complaint, priority_level
                    FROM patient_cases
                    WHERE case_id = %s
                    """,
                    (case_id,)
                )

                row = cursor.fetchone()

                if not row:
                    print(f"✗ Case {case_id} not found")
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

    def euclidean_distance(self,
                           case_id_a: int,
                           case_id_b: int) -> Optional[float]:

        vector_a = self.get_case_vector(case_id_a)
        vector_b = self.get_case_vector(case_id_b)

        if not vector_a or not vector_b:
            print("✗ One or both cases not found")
            return None

        diff = vector_a.to_numpy() - vector_b.to_numpy()
        distance = float(np.linalg.norm(diff))
        return distance

    def cosine_similarity(self,
                          case_id_a: int,
                          case_id_b: int) -> Optional[float]:

        vector_a = self.get_case_vector(case_id_a)
        vector_b = self.get_case_vector(case_id_b)

        if not vector_a or not vector_b:
            return None

        a = vector_a.to_numpy()
        b = vector_b.to_numpy()

        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        similarity = float(dot_product / (magnitude_a * magnitude_b))
        return similarity

    def get_all_vectors_as_matrix(self) -> Tuple[np.ndarray, List[int]]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT case_id, urgency_score, specialty_code,
                           time_intensity, stability_trend, distance_km
                    FROM patient_cases
                    WHERE current_status = 'Open'
                    ORDER BY case_id
                    """
                )
                rows = cursor.fetchall()

                if not rows:
                    return np.array([]), []

                case_ids = []
                vectors = []

                for row in rows:
                    case_ids.append(row['case_id'])
                    pv = PatientCaseVector(
                        urgency_score=row['urgency_score'],
                        specialty_code=row['specialty_code'],
                        time_intensity=float(row['time_intensity']),
                        stability_trend=float(row['stability_trend']),
                        distance_km=float(row['distance_km'])
                    )
                    vectors.append(pv.to_numpy())

                matrix = np.array(vectors)
                return matrix, case_ids

        finally:
            conn.close()

    def get_critical_cases(self) -> List[Dict]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT case_id, patient_id, chief_complaint,
                           urgency_score, specialty_code,
                           time_intensity, stability_trend,
                           distance_km, vector_magnitude, priority_level
                    FROM patient_cases
                    WHERE priority_level = 'CRITICAL'
                      AND current_status = 'Open'
                    ORDER BY urgency_score DESC, vector_magnitude DESC
                    """
                )
                return cursor.fetchall()

        finally:
            conn.close()


def main():
    db_config = {
        'host': 'localhost',
        'database': 'healthcomm',
        'user': 'postgres',
        'password': 'your_password_here',
        'port': 5432
    }

    print("\n" + "=" * 70)
    print("ONE-HOT ENCODED PATIENT CASE VECTOR DEMO")
    print("=" * 70)

    print("\n[ 1 ] ONE-HOT ENCODING DEMONSTRATION")
    print("-" * 40)

    for code, name in SPECIALTY_MAP.items():
        encoded = one_hot_encode_specialty(code)
        decoded = decode_one_hot_specialty(encoded)
        print(f"  Code {code} ({name:12s}) → {encoded} → decoded back to: {decoded}")

    print("\n[ 2 ] CREATING PATIENT CASE VECTORS")
    print("-" * 40)

    cases_data = [
        (9, 1, 4.5,  2.3, 3.2, "Acute MI"),
        (7, 3, 2.0,  0.5, 1.5, "Pediatric Seizure"),
        (4, 4, 1.0, -0.5, 8.0, "Tooth Abscess"),
        (3, 1, 0.5, -1.0, 5.0, "Cardiology Follow-up"),
        (8, 2, 3.0,  1.5, 2.0, "Asthma Attack")
    ]

    for urgency, specialty, time_i, trend, dist, name in cases_data:
        pv = PatientCaseVector(
            urgency_score=urgency,
            specialty_code=specialty,
            time_intensity=time_i,
            stability_trend=trend,
            distance_km=dist
        )
        print(f"\n  Case: {name}")
        print(f"  Raw input:   ({urgency}, {specialty}, {time_i}, {trend}, {dist})")
        print(f"  Encoded 8D:  {pv.to_numpy()}")
        print(f"  Magnitude:   {pv.magnitude():.4f}")

    print("\n[ 3 ] VECTOR MATHEMATICS COMPARISON")
    print("-" * 40)
    print("Comparing OLD (5D) vs NEW (8D one-hot) encoding:\n")

    old_cardiology = np.array([9.0, 1.0, 4.5, 2.3, 3.2])
    new_cardiology = PatientCaseVector(9, 1, 4.5, 2.3, 3.2).to_numpy()

    print(f"  OLD 5D vector: {old_cardiology}")
    print(f"  NEW 8D vector: {new_cardiology}")
    print(f"\n  OLD magnitude: {np.linalg.norm(old_cardiology):.4f}")
    print(f"  NEW magnitude: {np.linalg.norm(new_cardiology):.4f}")

    pv_cardiology = PatientCaseVector(9, 1, 4.5, 2.3, 3.2)
    pv_dental     = PatientCaseVector(9, 4, 4.5, 2.3, 3.2)

    old_diff_raw = abs(1 - 4)
    new_dist = float(np.linalg.norm(pv_cardiology.to_numpy() - pv_dental.to_numpy()))

    print(f"\n  [Distance between Cardiology and Dental cases]")
    print(f"  OLD encoding suggests distance = {old_diff_raw} (WRONG - implies ordering)")
    print(f"  NEW encoding gives distance    = {new_dist:.4f} (CORRECT - equal for all specialties)")

    dental_er_dist = float(np.linalg.norm(
        PatientCaseVector(9, 4, 4.5, 2.3, 3.2).to_numpy() -
        PatientCaseVector(9, 3, 4.5, 2.3, 3.2).to_numpy()
    ))
    print(f"  Dental vs ER distance          = {dental_er_dist:.4f} (same as above - all equal!)")
    print(f"  → One-hot treats ALL specialty differences as equal ✓")

    print("\n[ 4 ] VECTOR SCALER DEMONSTRATION")
    print("-" * 40)

    scaler = VectorScaler()

    print("\n  Min-Max scaling of individual features:\n")

    urgency_examples = [1, 5, 9, 10]
    for u in urgency_examples:
        scaled = scaler.scale_feature(u, 'urgency')
        print(f"  urgency {u:>2d} → scaled: {scaled:.3f}")

    print()

    stability_examples = [-10, -5, 0, 5, 10]
    print("  Stability trend scaling (negative=improving, positive=worsening):\n")
    for s in stability_examples:
        scaled = scaler.scale_feature(s, 'stability')
        direction = "improving" if s < 0 else ("stable" if s == 0 else "worsening")
        print(f"  stability {s:>4d} → scaled: {scaled:.3f}  ({direction})")

    print()

    print("  Clamping outliers beyond defined bounds:\n")
    outliers = [
        (150, 'distance', "150km clamped to 100km"),
        (-15, 'stability', "-15 clamped to -10"),
        (25,  'hours',    "25h clamped to 24h"),
        (0,   'urgency',  "0 urgency clamped to 1")
    ]

    for value, feature, explanation in outliers:
        scaled = scaler.scale_feature(value, feature)
        print(f"  {explanation:35s} → scaled: {scaled:.3f}")

    print("\n[ 5 ] FULL TRANSFORM PIPELINE")
    print("-" * 40)
    print("  transform_patient_vector() applies BOTH scaling AND one-hot\n")

    sample_cases = [
        {
            'urgency_score':   9,
            'specialty_code':  1,
            'time_intensity':  4.5,
            'stability_trend': 2.3,
            'distance_km':     3.2,
            '_label': 'Critical Cardiac'
        },
        {
            'urgency_score':   3,
            'specialty_code':  4,
            'time_intensity':  1.0,
            'stability_trend': -0.5,
            'distance_km':     8.0,
            '_label': 'Routine Dental'
        },
        {
            'urgency_score':   8,
            'specialty_code':  3,
            'time_intensity':  3.0,
            'stability_trend': 1.5,
            'distance_km':     2.0,
            '_label': 'ER Asthma Attack'
        }
    ]

    for case in sample_cases:
        label = case.pop('_label')

        raw_vector = PatientCaseVector(
            urgency_score   = case['urgency_score'],
            specialty_code  = case['specialty_code'],
            time_intensity  = case['time_intensity'],
            stability_trend = case['stability_trend'],
            distance_km     = case['distance_km']
        ).to_numpy()

        scaled_vector = scaler.transform_patient_vector(case)

        print(f"  {label}:")
        print(f"    Raw    (8D): {np.round(raw_vector, 3)}")
        print(f"    Scaled (8D): {np.round(scaled_vector, 3)}")
        print(f"    Raw  ||v||:  {np.linalg.norm(raw_vector):.4f}")
        print(f"    Scaled||v||: {np.linalg.norm(scaled_vector):.4f}")
        print()

    print("\n[ 6 ] WHY SCALING MATTERS FOR DISTANCE CALCULATIONS")
    print("-" * 40)

    case_a_raw = {'urgency_score': 9, 'specialty_code': 1,
                  'time_intensity': 4.5, 'stability_trend': 2.3, 'distance_km': 3.2}
    case_b_raw = {'urgency_score': 9, 'specialty_code': 1,
                  'time_intensity': 4.5, 'stability_trend': 2.3, 'distance_km': 53.2}
    case_c_raw = {'urgency_score': 4, 'specialty_code': 1,
                  'time_intensity': 4.5, 'stability_trend': 2.3, 'distance_km': 3.2}

    vec_a_unscaled = PatientCaseVector(**{k: v for k, v in case_a_raw.items()}).to_numpy()
    vec_b_unscaled = PatientCaseVector(**{k: v for k, v in case_b_raw.items()}).to_numpy()
    vec_c_unscaled = PatientCaseVector(**{k: v for k, v in case_c_raw.items()}).to_numpy()

    vec_a_scaled = scaler.transform_patient_vector(case_a_raw)
    vec_b_scaled = scaler.transform_patient_vector(case_b_raw)
    vec_c_scaled = scaler.transform_patient_vector(case_c_raw)

    dist_ab_unscaled = np.linalg.norm(vec_a_unscaled - vec_b_unscaled)
    dist_ac_unscaled = np.linalg.norm(vec_a_unscaled - vec_c_unscaled)
    dist_ab_scaled   = np.linalg.norm(vec_a_scaled   - vec_b_scaled)
    dist_ac_scaled   = np.linalg.norm(vec_a_scaled   - vec_c_scaled)

    print(f"\n  Case A vs Case B (50km distance difference):")
    print(f"    Unscaled distance: {dist_ab_unscaled:.4f}")
    print(f"    Scaled   distance: {dist_ab_scaled:.4f}")

    print(f"\n  Case A vs Case C (5-point urgency difference):")
    print(f"    Unscaled distance: {dist_ac_unscaled:.4f}")
    print(f"    Scaled   distance: {dist_ac_scaled:.4f}")

    print(f"\n  WITHOUT scaling: distance difference dominates urgency")
    print(f"    A-B: {dist_ab_unscaled:.2f} >> A-C: {dist_ac_unscaled:.2f}")
    print(f"    (50km gap appears {dist_ab_unscaled/dist_ac_unscaled:.1f}x bigger than 5-point urgency gap)")
    print(f"\n  WITH scaling: urgency difference correctly dominates")
    print(f"    A-B: {dist_ab_scaled:.4f} vs A-C: {dist_ac_scaled:.4f}")
    print(f"    → Scaling gives clinical features their proper weight ✓")

    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
