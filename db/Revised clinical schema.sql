-- Step 1: Enable vector support (if using pgvector, otherwise use REAL[])
-- CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE patient_cases (
    case_id SERIAL PRIMARY KEY,
    patient_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    current_status VARCHAR(20) DEFAULT 'Open',

    -- THE DENSE VECTOR (7 Dimensions)
    -- [urgency, spec_res_int, spec_crit_bias, spec_equip, time, stability, distance]
    case_vector REAL[] NOT NULL, 

    -- Metadata for human readability
    specialty_id INTEGER NOT NULL, -- FK to a specialties table
    chief_complaint TEXT,

    -- MIT STANDARD: Geometric Constraints
    -- Ensure the vector is exactly 7D (Adjust based on your final dimensions)
    CONSTRAINT vector_dimension_check CHECK (array_length(case_vector, 1) = 7),
    
    -- Ensure normalized values where appropriate (e.g., Urgency index 0)
    CONSTRAINT urgency_range_check CHECK (case_vector[1] BETWEEN 0.0 AND 10.0)
);

-- Step 2: High-Performance Indexing
-- If using pgvector: CREATE INDEX ON patient_cases USING ivfflat (case_vector vector_cosine_ops);
-- Using native arrays, we index the most queried metadata:
CREATE INDEX idx_priority_status ON patient_cases (current_status, specialty_id);
