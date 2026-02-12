CREATE TABLE patient_cases (
    -- 1. Identity Layer: Unique ID for every case
    case_id SERIAL PRIMARY KEY, -- Automatically increments (1, 2, 3...)
    patient_id INT NOT NULL,    -- Links to a patient table
    
    -- 2. Vector Dimensions: The 5 dimensions from your Python file
    urgency_score INT NOT NULL,   -- x1 (1-10)
    specialty_code INT NOT NULL,  -- x2 (1-4)
    time_intensity FLOAT NOT NULL, -- x3 (hours)
    stability_trend FLOAT NOT NULL, -- x4 (rate of change)
    distance_km FLOAT NOT NULL,    -- x5 (kilometers)
    
    -- 3. Descriptive Layer: The "Chief Complaint" and Status
    chief_complaint TEXT NOT NULL, -- Short description of illness
    priority_level VARCHAR(20),    -- e.g., 'CRITICAL', 'STANDARD'
    current_status VARCHAR(20) DEFAULT 'Open', -- e.g., 'Open', 'Closed'
    
    -- 4. Calculated Metadata
    vector_magnitude FLOAT,        -- Stores the pre-calculated Euclidean norm
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Track when the case was created
);
