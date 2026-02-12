#!/usr/bin/env python3

# ============================================================================
# HEALTHCOMM - PATIENT CASE VECTOR MANAGER (One-Hot Encoded Version)
# ============================================================================
#
# WHAT CHANGED FROM THE ORIGINAL:
#
# BEFORE (5-dimensional vector):
#   Patient_case = (x1, x2, x3, x4, x5)
#   x2 was a raw integer (1, 2, 3, or 4) representing specialty
#
# AFTER (8-dimensional vector):
#   Patient_case = (x1, s1, s2, s3, s4, x3, x4, x5)
#   x2 is now REPLACED by 4 binary columns (s1, s2, s3, s4)
#   Each sN is either 0 or 1
#
# WHY ONE-HOT ENCODING?
#   Raw integer encoding (1,2,3,4) creates a fake mathematical order:
#     - It suggests Dental (4) > ER (3) > Pediatrics (2) > Cardiology (1)
#     - ML models would wrongly treat these as if they have magnitude
#     - A model might think "Cardiology + Pediatrics = ER" which is nonsense
#
#   One-Hot encoding treats each specialty as a completely separate dimension:
#     Cardiology  → s1=1, s2=0, s3=0, s4=0
#     Pediatrics  → s1=0, s2=1, s3=0, s4=0
#     ER          → s1=0, s2=0, s3=1, s4=0
#     Dental      → s1=0, s2=0, s3=0, s4=1
#
#   This means:
#     - No specialty is "greater than" another
#     - Each specialty is independent
#     - ML models treat them correctly as categories
#
# FULL VECTOR LAYOUT:
#   Index: [0]  [1]  [2]  [3]  [4]  [5]   [6]   [7]
#   Field:  x1   s1   s2   s3   s4   x3    x4    x5
#   Meaning: urgency, cardio, peds, ER, dental, time, trend, distance
# ============================================================================


# --- IMPORTS ---

import psycopg2
# psycopg2 is the PostgreSQL adapter for Python
# It allows us to connect to and query PostgreSQL databases
# "import psycopg2" loads the library into memory so we can use it

from psycopg2.extras import RealDictCursor
# RealDictCursor is a special cursor type from psycopg2
# Normally, query results come back as tuples: (1, 'Dr. Chen', 'Doctor')
# RealDictCursor makes them come back as dictionaries: {'user_id': 1, 'username': 'Dr. Chen'}
# This makes the code more readable since we access values by name, not index

import numpy as np
# numpy is Python's core numerical computing library
# "np" is the conventional shorthand alias everyone uses
# We need it for:
#   - np.array(): creating arrays/vectors
#   - np.zeros(): creating arrays filled with zeros (key for one-hot encoding)
#   - np.linalg.norm(): calculating vector magnitude
#   - np.dot(): computing dot products

from typing import List, Dict, Optional, Tuple
# These are type hints imported from Python's typing module
# They don't affect how code runs, but help with:
#   - Code readability (you know what type a function expects/returns)
#   - IDE autocomplete and error detection
# List[int] = a list of integers
# Dict[str, str] = a dictionary with string keys and string values
# Optional[int] = either an int OR None (value might not exist)
# Tuple = a fixed-length, immutable sequence of values

from dataclasses import dataclass, field
# dataclass is a Python decorator that auto-generates common class methods
# @dataclass automatically creates: __init__, __repr__, __eq__
# Without it, we'd have to manually write: def __init__(self, urgency_score, ...)
# 'field' lets us customize dataclass field behavior (e.g., default values)


# ============================================================================
# SPECIALTY MAPPING - The foundation of One-Hot Encoding
# ============================================================================

SPECIALTY_MAP = {
    # This is a Python dictionary (key-value store)
    # Keys are integers (1, 2, 3, 4) representing specialty codes
    # Values are strings (human-readable specialty names)
    # We use this to:
    #   1. Validate that a specialty code is valid (must be a key in this dict)
    #   2. Display human-readable names in output
    #   3. Determine the position of the '1' in the one-hot vector

    1: "Cardiology",     # Code 1 → position 0 in one-hot vector (s1)
    2: "Pediatrics",     # Code 2 → position 1 in one-hot vector (s2)
    3: "ER",             # Code 3 → position 2 in one-hot vector (s3)
    4: "Dental"          # Code 4 → position 3 in one-hot vector (s4)
}

NUM_SPECIALTIES = len(SPECIALTY_MAP)
# len() returns the number of items in a collection
# len(SPECIALTY_MAP) = 4 (since there are 4 specialties)
# We store this as a constant so we never hardcode "4" anywhere
# If we add a 5th specialty later, only SPECIALTY_MAP needs updating


# ============================================================================
# ONE-HOT ENCODING FUNCTION
# ============================================================================

def one_hot_encode_specialty(specialty_code: int) -> np.ndarray:
    # This is a standalone function (not inside a class)
    # It takes one input: specialty_code (must be an integer)
    # It returns: np.ndarray (a numpy array of 4 elements)
    #
    # The ': int' and '-> np.ndarray' are TYPE HINTS
    # They say "this parameter should be an int" and "this returns a numpy array"
    # Python doesn't enforce them, but they document intent clearly

    """
    Convert a specialty code integer into a one-hot encoded vector.

    Examples:
        one_hot_encode_specialty(1) → [1, 0, 0, 0]  (Cardiology)
        one_hot_encode_specialty(2) → [0, 1, 0, 0]  (Pediatrics)
        one_hot_encode_specialty(3) → [0, 0, 1, 0]  (ER)
        one_hot_encode_specialty(4) → [0, 0, 0, 1]  (Dental)
    """

    if specialty_code not in SPECIALTY_MAP:
        # Check if the given code exists in our dictionary
        # 'not in' is Python's membership test operator
        # If specialty_code is not a key in SPECIALTY_MAP, raise an error
        # Example: specialty_code=5 would trigger this since only 1,2,3,4 exist
        raise ValueError(
            f"Invalid specialty code: {specialty_code}. "
            f"Must be one of {list(SPECIALTY_MAP.keys())}"
            # f"..." is an f-string (formatted string)
            # It evaluates {specialty_code} and {list(...)} at runtime
            # list(SPECIALTY_MAP.keys()) converts dict keys to: [1, 2, 3, 4]
        )

    one_hot = np.zeros(NUM_SPECIALTIES, dtype=int)
    # np.zeros() creates an array filled entirely with zeros
    # np.zeros(4, dtype=int) → array([0, 0, 0, 0])
    # dtype=int ensures values are integers (0 and 1), not floats (0.0, 1.0)
    # NUM_SPECIALTIES=4 sets the length of the array
    # At this point: one_hot = [0, 0, 0, 0]

    one_hot[specialty_code - 1] = 1
    # We set the element at position (specialty_code - 1) to 1
    # We subtract 1 because:
    #   - specialty_code starts at 1 (not 0)
    #   - Python array indices start at 0
    # So:
    #   specialty_code=1 → index 0 → one_hot[0] = 1 → [1, 0, 0, 0]
    #   specialty_code=2 → index 1 → one_hot[1] = 1 → [0, 1, 0, 0]
    #   specialty_code=3 → index 2 → one_hot[2] = 1 → [0, 0, 1, 0]
    #   specialty_code=4 → index 3 → one_hot[3] = 1 → [0, 0, 0, 1]

    return one_hot
    # Return the completed one-hot array to the caller


def decode_one_hot_specialty(one_hot: np.ndarray) -> int:
    # This is the REVERSE function - converts one-hot vector back to integer code
    # Takes a numpy array as input
    # Returns the specialty code integer

    """
    Convert a one-hot encoded specialty vector back to a specialty code.

    Examples:
        decode_one_hot_specialty([1, 0, 0, 0]) → 1  (Cardiology)
        decode_one_hot_specialty([0, 0, 1, 0]) → 3  (ER)
    """

    index = np.argmax(one_hot)
    # np.argmax() finds the INDEX of the maximum value in an array
    # Since our one-hot has exactly one '1' and the rest are '0':
    #   np.argmax([1, 0, 0, 0]) → 0
    #   np.argmax([0, 1, 0, 0]) → 1
    #   np.argmax([0, 0, 1, 0]) → 2
    #   np.argmax([0, 0, 0, 1]) → 3

    return int(index + 1)
    # Add 1 to convert from 0-based index back to 1-based specialty code
    # int() converts numpy int64 to regular Python int (cleaner output)
    # index=0 → specialty_code=1 (Cardiology)
    # index=2 → specialty_code=3 (ER)


# ============================================================================
# PATIENT CASE VECTOR DATACLASS
# ============================================================================

@dataclass
# @dataclass is a DECORATOR - it modifies the class below it
# It automatically generates:
#   __init__: allows PatientCaseVector(urgency_score=9, specialty_code=1, ...)
#   __repr__: but we override this with our own __repr__ below
#   __eq__: allows comparing two PatientCaseVectors with ==
class PatientCaseVector:

    """
    Represents a patient case as an 8-dimensional one-hot encoded vector.

    Internal storage:    (x1, specialty_code, x3, x4, x5)  ← raw values
    Encoded vector:      (x1, s1, s2, s3, s4, x3, x4, x5) ← 8D numpy array

    Dimensions:
      [0] x1 = urgency_score    (integer 1-10)
      [1] s1 = is_cardiology    (0 or 1)
      [2] s2 = is_pediatrics    (0 or 1)
      [3] s3 = is_er            (0 or 1)
      [4] s4 = is_dental        (0 or 1)
      [5] x3 = time_intensity   (float, hours)
      [6] x4 = stability_trend  (float, negative=improving, positive=worsening)
      [7] x5 = distance_km      (float, kilometers)
    """

    # --- REQUIRED FIELDS (must be provided when creating the object) ---

    urgency_score: int
    # The : int is a TYPE ANNOTATION (not enforcement, just documentation)
    # This is x1 in our vector - stored as a plain integer (1-10)
    # The @dataclass decorator reads this and auto-includes it in __init__

    specialty_code: int
    # This is x2 - stored as integer (1-4) for database storage convenience
    # When we generate the numpy vector, this gets ONE-HOT ENCODED
    # We keep the raw integer because that's what the database stores

    time_intensity: float
    # This is x3 - stored as a Python float
    # Represents hours of care required (can be fractional like 2.5)

    stability_trend: float
    # This is x4 - stored as a Python float
    # SIGN MATTERS: negative = improving, positive = worsening
    # Example: -1.5 means vitals improving at rate 1.5 per hour

    distance_km: float
    # This is x5 - stored as a Python float
    # Distance in kilometers from patient to provider

    # --- OPTIONAL FIELDS (have default values, can be omitted) ---

    case_id: Optional[int] = None
    # Optional[int] means this can be int OR None
    # Default is None (no value) because new cases don't have an ID yet
    # The database assigns the ID after INSERT

    patient_id: Optional[int] = None
    # Links this case to a patient in the patients table
    # Optional because we might create the vector object before saving to DB

    chief_complaint: Optional[str] = None
    # Brief description of why the patient presented
    # Optional[str] = either a string or None

    priority_level: Optional[str] = None
    # Auto-computed by the database (CRITICAL/HIGH/MEDIUM/LOW)
    # None until retrieved from database

    # -------------------------------------------------------------------------
    def __post_init__(self):
        # __post_init__ is a special method called BY @dataclass AFTER __init__
        # We use it to run validation AFTER all fields are set
        # If we put validation in __init__, @dataclass would override it
        """Validate all vector components after initialization."""

        if not 1 <= self.urgency_score <= 10:
            # '1 <= self.urgency_score <= 10' is Python's CHAINED COMPARISON
            # Equivalent to: (1 <= self.urgency_score) AND (self.urgency_score <= 10)
            # If urgency is outside 1-10, this condition is True → raise error
            raise ValueError(
                f"Urgency score must be 1-10, got: {self.urgency_score}"
            )

        if self.specialty_code not in SPECIALTY_MAP:
            # Check if specialty_code is a valid key in SPECIALTY_MAP
            # 'not in' tests membership - True if the key doesn't exist
            raise ValueError(
                f"Specialty code must be 1-4, got: {self.specialty_code}"
            )

        if self.time_intensity < 0:
            # Time cannot be negative - you can't have -2 hours of care
            raise ValueError(
                f"Time intensity must be >= 0, got: {self.time_intensity}"
            )

        if self.distance_km < 0:
            # Distance cannot be negative - you can't be -5km away
            raise ValueError(
                f"Distance must be >= 0, got: {self.distance_km}"
            )

    # -------------------------------------------------------------------------
    def get_one_hot_specialty(self) -> np.ndarray:
        """Return the one-hot encoded specialty as a numpy array."""
        return one_hot_encode_specialty(self.specialty_code)
        # Simply calls our standalone one_hot_encode_specialty function
        # This is an instance method (belongs to the object)
        # 'self' refers to the current PatientCaseVector object
        # Example: if self.specialty_code = 3 (ER), returns [0, 0, 1, 0]

    # -------------------------------------------------------------------------
    def to_numpy(self) -> np.ndarray:
        """
        Convert the patient case to an 8-dimensional numpy array.

        This is the CORE METHOD that applies One-Hot Encoding.

        Vector layout:
          [0]      [1][2][3][4]        [5]           [6]              [7]
          urgency  ← one-hot →  time_intensity  stability_trend  distance_km
        """

        one_hot = self.get_one_hot_specialty()
        # Call our method to get the 4-element one-hot array
        # If specialty is Cardiology (1): one_hot = [1, 0, 0, 0]
        # If specialty is ER (3): one_hot = [0, 0, 1, 0]

        vector = np.array([
            # np.array() creates a numpy array from a Python list
            # Every element must be a number (int or float)
            # numpy converts everything to a common type automatically

            float(self.urgency_score),
            # float() converts the integer to a float
            # We use float throughout for mathematical consistency
            # Without this, numpy might create an int array, which limits math operations

            float(one_hot[0]),
            # one_hot[0] is s1: is_cardiology (1 if cardiology, else 0)
            # Array indexing: one_hot[0] accesses the first element

            float(one_hot[1]),
            # one_hot[1] is s2: is_pediatrics (1 if pediatrics, else 0)

            float(one_hot[2]),
            # one_hot[2] is s3: is_er (1 if ER, else 0)

            float(one_hot[3]),
            # one_hot[3] is s4: is_dental (1 if dental, else 0)

            float(self.time_intensity),
            # x3: hours of care required - already a float

            float(self.stability_trend),
            # x4: rate of change in vitals - already a float
            # Can be negative (improving) or positive (worsening)

            float(self.distance_km)
            # x5: distance in kilometers - already a float
        ])

        return vector
        # Returns an 8-element numpy array like:
        # array([9., 1., 0., 0., 0., 4.5, 2.3, 3.2])
        #        ↑   ↑              ↑    ↑    ↑
        #       urg  card(yes)    time trend dist

    # -------------------------------------------------------------------------
    def magnitude(self) -> float:
        """
        Calculate the Euclidean magnitude (norm) of the 8D vector.

        Formula: ||v|| = sqrt(x1² + s1² + s2² + s3² + s4² + x3² + x4² + x5²)

        Since s values are 0 or 1: s² = s (e.g. 1²=1, 0²=0)
        So one-hot entries contribute exactly 1 or 0 to the sum.
        """
        return float(np.linalg.norm(self.to_numpy()))
        # np.linalg.norm() computes the Euclidean norm of a vector
        # Formula: sqrt(sum of squares of all elements)
        # float() converts numpy float64 to Python float for clean output
        # self.to_numpy() gets the 8D vector first, then norm is computed

    # -------------------------------------------------------------------------
    def specialty_name(self) -> str:
        """Return the human-readable name of the specialty."""
        return SPECIALTY_MAP[self.specialty_code]
        # Dictionary lookup: SPECIALTY_MAP[1] → "Cardiology"
        # self.specialty_code gives us the integer key
        # SPECIALTY_MAP[key] returns the string value

    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        # __repr__ is Python's "representation" method
        # Called when you print() an object or view it in the console
        # We override @dataclass's auto-generated one for cleaner output
        """Return a human-readable string representation."""

        one_hot = self.get_one_hot_specialty()
        # Get the 4-element one-hot array for display
        # e.g., [1, 0, 0, 0] for Cardiology

        one_hot_str = f"[{','.join(str(int(x)) for x in one_hot)}]"
        # Build a string like "[1,0,0,0]" from the array
        # str(int(x)) converts each float (1.0, 0.0) to int string ("1", "0")
        # ','.join([...]) joins list elements with commas: "1,0,0,0"
        # The outer f"[...]" adds the square brackets

        return (
            f"PatientCaseVector(\n"
            # f-string with \n for newline - creates multi-line output
            f"  x1 (urgency)    = {self.urgency_score}/10\n"
            f"  x2 (specialty)  = {one_hot_str} → {self.specialty_name()}\n"
            # Shows both the encoded form AND the human name
            f"  x3 (time)       = {self.time_intensity:.1f} hours\n"
            # :.1f formats float to 1 decimal place (e.g., 4.5)
            f"  x4 (trend)      = {self.stability_trend:+.2f}\n"
            # :+.2f shows the sign explicitly (+2.30 or -1.50)
            f"  x5 (distance)   = {self.distance_km:.1f} km\n"
            f"  full vector     = {self.to_numpy()}\n"
            # Shows complete 8D numpy array
            f"  ||v||           = {self.magnitude():.4f}\n"
            # :.4f formats to 4 decimal places for precision
            f"  priority        = {self.priority_level or 'Not computed'}\n"
            # 'or' returns right side if left side is falsy (None → 'Not computed')
            f")"
        )


# ============================================================================
# PATIENT CASE VECTOR MANAGER CLASS
# ============================================================================

class PatientCaseVectorManager:
    """
    Manages patient case vectors with one-hot encoded specialties.
    Handles database operations and vector mathematics.
    """

    VECTOR_DIMENSION = 8
    # Class-level constant (shared by all instances)
    # Reminds us and documents that vectors are now 8-dimensional
    # Before one-hot encoding, this was 5
    # Format: [urgency, s1, s2, s3, s4, time, trend, distance]

    # -------------------------------------------------------------------------
    def __init__(self, db_config: Dict[str, str]):
        # __init__ is the constructor - called when creating a new instance
        # 'self' is the instance being created
        # db_config: Dict[str, str] means a dictionary with string keys and values
        """Initialize with database connection configuration."""

        self.db_config = db_config
        # Store db_config as an instance attribute
        # 'self.db_config' makes it accessible to all other methods in the class
        # Without 'self.', it would be a local variable that disappears after __init__

    # -------------------------------------------------------------------------
    def get_connection(self):
        """Create and return a new database connection."""
        return psycopg2.connect(**self.db_config)
        # psycopg2.connect() opens a connection to PostgreSQL
        # **self.db_config unpacks the dictionary as keyword arguments
        # So {'host': 'localhost', 'database': 'healthcomm', ...}
        # becomes: psycopg2.connect(host='localhost', database='healthcomm', ...)
        # We create a new connection each time (simple approach)

    # -------------------------------------------------------------------------
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
        """
        Create a new patient case and save to database.

        The specialty_code is stored as integer in DB,
        but exposed as one-hot when converted to numpy vector.

        Returns:
            case_id (int) if successful, None if failed
        """

        vector = PatientCaseVector(
            # Instantiate a PatientCaseVector to VALIDATE all inputs
            # before touching the database
            # If any value is invalid, __post_init__ raises ValueError here
            # and we never reach the database code
            urgency_score=urgency_score,
            specialty_code=specialty_code,
            time_intensity=time_intensity,
            stability_trend=stability_trend,
            distance_km=distance_km
        )

        conn = self.get_connection()
        # Open a database connection
        # 'conn' is a psycopg2 connection object

        try:
            # try/except/finally ensures we handle errors properly
            # Code in 'try' runs normally
            # Code in 'except' runs if an error occurs
            # Code in 'finally' ALWAYS runs (cleanup)

            with conn.cursor() as cursor:
                # 'with' is a context manager - auto-closes cursor when done
                # conn.cursor() creates a database cursor
                # A cursor is the object we use to execute SQL queries

                cursor.execute(
                    # execute() sends a SQL query to the database
                    """
                    INSERT INTO patient_cases (
                        patient_id, created_by_user_id,
                        urgency_score, specialty_code, time_intensity,
                        stability_trend, distance_km,
                        chief_complaint, case_description
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING case_id, priority_level, vector_magnitude
                    """,
                    # %s are PARAMETERIZED PLACEHOLDERS
                    # Never use f-strings for SQL - risk of SQL injection!
                    # psycopg2 safely escapes the values we pass

                    (patient_id, created_by_user_id,
                     urgency_score, specialty_code,
                     time_intensity, stability_trend,
                     distance_km, chief_complaint, case_description)
                    # Tuple of values - maps to %s placeholders in order
                    # Note: specialty_code is stored as integer in the database
                    # One-hot encoding is applied in Python, not SQL
                )

                result = cursor.fetchone()
                # fetchone() retrieves ONE row from the query result
                # RETURNING clause makes INSERT return the specified columns
                # result is a tuple: (case_id, priority_level, vector_magnitude)

                case_id, priority, magnitude = result
                # TUPLE UNPACKING - assign each element to a variable
                # case_id gets result[0], priority gets result[1], etc.
                # Equivalent to:
                #   case_id = result[0]
                #   priority = result[1]
                #   magnitude = result[2]

                conn.commit()
                # commit() saves the transaction to the database permanently
                # Without commit(), the INSERT would be rolled back
                # Think of it as "Save" in a document editor

                print(f"✓ Case {case_id} created!")
                print(f"  One-Hot Vector: {vector.to_numpy()}")
                # Show the full 8D encoded vector
                print(f"  Specialty: {vector.specialty_name()} "
                      f"→ One-Hot: {vector.get_one_hot_specialty()}")
                # Show what specialty maps to which one-hot bits
                print(f"  Priority: {priority}, ||v||={float(magnitude):.4f}")
                # float() converts database Decimal type to Python float

                return case_id
                # Return the new case's ID so caller can reference it

        except ValueError as e:
            # Catch validation errors specifically
            # These come from PatientCaseVector.__post_init__
            print(f"✗ Validation error: {e}")
            return None

        except psycopg2.Error as e:
            # Catch any database-specific errors
            # psycopg2.Error is the base class for all psycopg2 exceptions
            conn.rollback()
            # rollback() undoes any uncommitted changes
            # If INSERT fails halfway, rollback cleans up partial state
            print(f"✗ Database error: {e}")
            return None

        finally:
            conn.close()
            # Always close the connection regardless of success or failure
            # Prevents connection leaks (too many open connections crash the DB)

    # -------------------------------------------------------------------------
    def get_case_vector(self, case_id: int) -> Optional[PatientCaseVector]:
        """
        Retrieve a case from the database and return as PatientCaseVector.
        The specialty_code from DB is automatically one-hot encoded
        when to_numpy() is called on the returned object.
        """

        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # cursor_factory=RealDictCursor makes rows return as dicts
                # So we access row['urgency_score'] instead of row[0]

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
                    # Note: (case_id,) is a ONE-ELEMENT TUPLE
                    # The comma is required - without it, (case_id) is just parentheses
                    # psycopg2 requires parameters as a tuple or list
                )

                row = cursor.fetchone()
                # fetchone() returns one row as a dictionary (due to RealDictCursor)
                # Returns None if no row found

                if not row:
                    # 'not row' is True when row is None (case not found)
                    print(f"✗ Case {case_id} not found")
                    return None

                return PatientCaseVector(
                    # Construct and return a PatientCaseVector from the database row
                    # The specialty_code is stored as int in DB
                    # One-hot encoding happens automatically when to_numpy() is called

                    case_id=row['case_id'],
                    # row['case_id'] accesses the 'case_id' column value
                    # This works because we used RealDictCursor

                    patient_id=row['patient_id'],

                    urgency_score=row['urgency_score'],
                    # Database returns an integer - matches our int type

                    specialty_code=row['specialty_code'],
                    # Database stores integer (1, 2, 3, or 4)
                    # one_hot_encode_specialty() will handle conversion

                    time_intensity=float(row['time_intensity']),
                    # float() converts PostgreSQL Decimal to Python float
                    # Required because psycopg2 returns DECIMAL as Decimal objects

                    stability_trend=float(row['stability_trend']),
                    # Same conversion for stability trend

                    distance_km=float(row['distance_km']),
                    # Same conversion for distance

                    chief_complaint=row['chief_complaint'],
                    # String - no conversion needed

                    priority_level=row['priority_level']
                    # String - no conversion needed (CRITICAL/HIGH/MEDIUM/LOW)
                )

        finally:
            conn.close()
            # Always close, even if an exception was raised

    # -------------------------------------------------------------------------
    def euclidean_distance(self,
                           case_id_a: int,
                           case_id_b: int) -> Optional[float]:
        """
        Calculate Euclidean distance between two 8D case vectors.

        Formula: d = ||v_a - v_b|| = sqrt(sum((v_a[i] - v_b[i])^2))

        Uses one-hot encoded vectors, so specialty differences are:
          - Same specialty: 0 contribution to distance
          - Different specialty: sqrt(2) contribution (since two bits flip)
        """

        vector_a = self.get_case_vector(case_id_a)
        # Retrieve first case from DB as PatientCaseVector object

        vector_b = self.get_case_vector(case_id_b)
        # Retrieve second case from DB as PatientCaseVector object

        if not vector_a or not vector_b:
            # 'not vector_a' is True if vector_a is None (case not found)
            # 'or' means: if EITHER is missing, we can't compute distance
            print("✗ One or both cases not found")
            return None

        diff = vector_a.to_numpy() - vector_b.to_numpy()
        # Numpy ARRAY SUBTRACTION - subtracts element-by-element
        # vector_a.to_numpy() returns 8D array
        # vector_b.to_numpy() returns 8D array
        # Subtraction: [a0-b0, a1-b1, a2-b2, ..., a7-b7]
        # Example:
        #   v_a = [9, 1, 0, 0, 0, 4.5,  2.3, 3.2]  (Cardiology)
        #   v_b = [7, 0, 0, 1, 0, 2.0,  0.5, 1.5]  (ER)
        #   diff= [2, 1, 0,-1, 0, 2.5,  1.8, 1.7]

        distance = float(np.linalg.norm(diff))
        # np.linalg.norm() computes the Euclidean norm of the diff vector
        # This is: sqrt(2² + 1² + 0² + (-1)² + 0² + 2.5² + 1.8² + 1.7²)
        # = sqrt(4 + 1 + 0 + 1 + 0 + 6.25 + 3.24 + 2.89)
        # = sqrt(18.38) ≈ 4.29
        # float() converts numpy float64 → Python float

        return distance

    # -------------------------------------------------------------------------
    def cosine_similarity(self,
                          case_id_a: int,
                          case_id_b: int) -> Optional[float]:
        """
        Calculate cosine similarity between two 8D case vectors.

        Formula: cos(θ) = (v_a · v_b) / (||v_a|| × ||v_b||)

        Range: 0 to 1 for our vectors (all positive/zero values)
          1.0 = identical direction (very similar cases)
          0.0 = perpendicular (completely different case profiles)
        """

        vector_a = self.get_case_vector(case_id_a)
        vector_b = self.get_case_vector(case_id_b)

        if not vector_a or not vector_b:
            return None

        a = vector_a.to_numpy()
        # Get 8D numpy array for case A

        b = vector_b.to_numpy()
        # Get 8D numpy array for case B

        dot_product = np.dot(a, b)
        # np.dot() computes the DOT PRODUCT of two vectors
        # dot(a, b) = sum(a[i] * b[i]) for all i
        # = a[0]*b[0] + a[1]*b[1] + ... + a[7]*b[7]
        # High dot product = vectors point in similar direction

        magnitude_a = np.linalg.norm(a)
        # Compute ||v_a|| (magnitude of vector A)

        magnitude_b = np.linalg.norm(b)
        # Compute ||v_b|| (magnitude of vector B)

        if magnitude_a == 0 or magnitude_b == 0:
            # Guard against division by zero
            # A zero vector has no direction, similarity is undefined
            return 0.0

        similarity = float(dot_product / (magnitude_a * magnitude_b))
        # COSINE SIMILARITY formula
        # Dividing by the product of magnitudes NORMALIZES the result
        # This means result is independent of vector scale
        # Only direction matters: two cases with identical PROFILES
        # but different SCALES still get similarity = 1.0

        return similarity

    # -------------------------------------------------------------------------
    def get_all_vectors_as_matrix(self) -> Tuple[np.ndarray, List[int]]:
        """
        Retrieve all case vectors from DB as a 2D numpy matrix.

        Returns:
            (matrix, case_ids) where:
            - matrix shape: (num_cases, 8)
            - case_ids: list of case IDs in matching row order

        Useful for:
            - Batch ML operations
            - Clustering all cases together
            - Computing pairwise distances efficiently
        """

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
                # fetchall() retrieves ALL rows at once as a list of dicts

                if not rows:
                    # Return empty arrays if no cases found
                    return np.array([]), []

                case_ids = []
                # Initialize empty list to store case IDs in order

                vectors = []
                # Initialize empty list to store 8D numpy arrays

                for row in rows:
                    # Iterate over each database row
                    # 'row' is a RealDictCursor dictionary each iteration

                    case_ids.append(row['case_id'])
                    # .append() adds an item to the end of the list

                    pv = PatientCaseVector(
                        # Create a PatientCaseVector for each row
                        urgency_score=row['urgency_score'],
                        specialty_code=row['specialty_code'],
                        time_intensity=float(row['time_intensity']),
                        stability_trend=float(row['stability_trend']),
                        distance_km=float(row['distance_km'])
                    )

                    vectors.append(pv.to_numpy())
                    # Convert to 8D one-hot encoded vector and add to list
                    # After the loop, vectors is a list of 8-element arrays

                matrix = np.array(vectors)
                # np.array() converts a list of arrays into a 2D MATRIX
                # If vectors = [[v1], [v2], [v3]], matrix shape is (3, 8)
                # matrix[0] = first case's 8D vector
                # matrix[:, 0] = urgency scores of ALL cases (column 0)

                return matrix, case_ids
                # Return both the matrix and the ID list
                # Caller needs the IDs to know which row corresponds to which case

        finally:
            conn.close()

    # -------------------------------------------------------------------------
    def get_critical_cases(self) -> List[Dict]:
        """Get all open CRITICAL priority cases with their vectors."""

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
                # Returns list of RealDictCursor rows
                # Each row is a dict: {'case_id': 1, 'urgency_score': 9, ...}

        finally:
            conn.close()


# ============================================================================
# DEMONSTRATION FUNCTION
# ============================================================================

def main():
    """
    Demonstrate the one-hot encoded patient case vector system.
    Shows encoding, vector operations, and similarity calculations.
    """

    db_config = {
        # Database connection settings
        'host': 'localhost',       # PostgreSQL server address
        'database': 'healthcomm', # Database name
        'user': 'postgres',       # PostgreSQL username
        'password': 'your_password_here', # Replace with real password
        'port': 5432              # Default PostgreSQL port
    }

    print("\n" + "=" * 70)
    print("ONE-HOT ENCODED PATIENT CASE VECTOR DEMO")
    print("=" * 70)

    # --- SECTION 1: ONE-HOT ENCODING DEMONSTRATION ---
    print("\n[ 1 ] ONE-HOT ENCODING DEMONSTRATION")
    print("-" * 40)

    for code, name in SPECIALTY_MAP.items():
        # Iterate over all specialty codes
        # .items() returns (key, value) pairs: (1, 'Cardiology'), etc.

        encoded = one_hot_encode_specialty(code)
        # Get one-hot vector for this specialty

        decoded = decode_one_hot_specialty(encoded)
        # Decode it back to integer (should match original code)

        print(f"  Code {code} ({name:12s}) → {encoded} → decoded back to: {decoded}")
        # {:12s} pads the name string to 12 characters for alignment

    # --- SECTION 2: VECTOR CREATION ---
    print("\n[ 2 ] CREATING PATIENT CASE VECTORS")
    print("-" * 40)

    cases_data = [
        # List of tuples: (urgency, specialty, time, trend, distance, name)
        (9, 1, 4.5,  2.3, 3.2, "Acute MI"),
        (7, 3, 2.0,  0.5, 1.5, "Pediatric Seizure"),
        (4, 4, 1.0, -0.5, 8.0, "Tooth Abscess"),
        (3, 1, 0.5, -1.0, 5.0, "Cardiology Follow-up"),
        (8, 2, 3.0,  1.5, 2.0, "Asthma Attack")
    ]

    for urgency, specialty, time_i, trend, dist, name in cases_data:
        # TUPLE UNPACKING in a for loop
        # Each tuple in cases_data is unpacked into these variables

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

    # --- SECTION 3: VECTOR MATH COMPARISON ---
    print("\n[ 3 ] VECTOR MATHEMATICS COMPARISON")
    print("-" * 40)
    print("Comparing OLD (5D) vs NEW (8D one-hot) encoding:\n")

    # Old 5D vector (raw specialty integer)
    old_cardiology = np.array([9.0, 1.0, 4.5, 2.3, 3.2])
    # New 8D vector (one-hot specialty)
    new_cardiology = PatientCaseVector(9, 1, 4.5, 2.3, 3.2).to_numpy()

    print(f"  OLD 5D vector: {old_cardiology}")
    print(f"  NEW 8D vector: {new_cardiology}")
    print(f"\n  OLD magnitude: {np.linalg.norm(old_cardiology):.4f}")
    print(f"  NEW magnitude: {np.linalg.norm(new_cardiology):.4f}")

    # Show why encoding matters for distance calculation
    pv_cardiology = PatientCaseVector(9, 1, 4.5, 2.3, 3.2)
    # Cardiology case: one-hot = [1, 0, 0, 0]

    pv_dental = PatientCaseVector(9, 4, 4.5, 2.3, 3.2)
    # Dental case: one-hot = [0, 0, 0, 1]
    # IDENTICAL in all dimensions except specialty!

    old_diff_raw = abs(1 - 4)
    # Old encoding: |Cardiology - Dental| = |1 - 4| = 3
    # This is WRONG - implies Dental is "3 units away" from Cardiology

    new_dist = float(np.linalg.norm(
        pv_cardiology.to_numpy() - pv_dental.to_numpy()
    ))
    # New encoding: distance between [1,0,0,0] and [0,0,0,1]
    # = sqrt((1-0)^2 + (0-0)^2 + (0-0)^2 + (0-1)^2)
    # = sqrt(1 + 0 + 0 + 1) = sqrt(2) ≈ 1.414
    # This is CORRECT - all specialties are EQUALLY different from each other

    print(f"\n  [Distance between Cardiology and Dental cases]")
    print(f"  OLD encoding suggests distance = {old_diff_raw} (WRONG - implies ordering)")
    print(f"  NEW encoding gives distance    = {new_dist:.4f} (CORRECT - equal for all specialties)")

    dental_er_dist = float(np.linalg.norm(
        PatientCaseVector(9, 4, 4.5, 2.3, 3.2).to_numpy() -
        PatientCaseVector(9, 3, 4.5, 2.3, 3.2).to_numpy()
    ))
    # Distance between Dental and ER with same other components
    print(f"  Dental vs ER distance          = {dental_er_dist:.4f} (same as above - all equal!)")
    print(f"  → One-hot treats ALL specialty differences as equal ✓")

    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    # This block only runs when this file is executed directly
    # NOT when it's imported as a module by another file
    # Standard Python pattern for runnable scripts
    main()
