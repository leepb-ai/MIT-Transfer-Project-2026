-- MIT Transfer Project 2026: Clinical Case Schema
-- Optimized for 8D Vector Space & One-Hot Encoding

CREATE TABLE patient_cases (
    case_id SERIAL PRIMARY KEY,
    patient_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 1. Urgency Score [0,1]
    urgency_score NUMERIC(3,2) NOT NULL CHECK (urgency_score BETWEEN 0.00 AND 1.00),

    -- 2. One-Hot Encoded Specialties (Exactly one must be 1.0)
    -- Using numeric(2,1) to store 0.0 or 1.0
    spec_cardiology   NUMERIC(2,1) DEFAULT 0.0 CHECK (spec_cardiology IN (0.0, 1.0)),
    spec_pediatrics   NUMERIC(2,1) DEFAULT 0.0 CHECK (spec_pediatrics IN (0.0, 1.0)),
    spec_neurology    NUMERIC(2,1) DEFAULT 0.0 CHECK (spec_neurology IN (0.0, 1.0)),
    spec_orthopedics  NUMERIC(2,1) DEFAULT 0.0 CHECK (spec_orthopedics IN (0.0, 1.0)),

    -- 3. Time Intensity [0,1]
    time_intensity NUMERIC(3,2) NOT NULL CHECK (time_intensity BETWEEN 0.00 AND 1.00),

    -- 4. Clinical Stability [0,1] (1.0 is stable, 0.0 is crashing)
    stability_score NUMERIC(3,2) NOT NULL CHECK (stability_score BETWEEN 0.00 AND 1.00),

    -- 5. Physical Distance [0,1] (Normalized)
    location_distance NUMERIC(3,2) NOT NULL CHECK (location_distance BETWEEN 0.00 AND 1.00),

    -- MIT STANDARD: Integrity Constraint
    -- This ensures the One-Hot encoding is valid (exactly one specialty selected)
    CONSTRAINT one_hot_specialty_check 
    CHECK ((spec_cardiology + spec_pediatrics + spec_neurology + spec_orthopedics) = 1.0)
);

-- Index for faster retrieval by the Vector Manager
CREATE INDEX idx_case_urgency ON patient_cases (urgency_score DESC);