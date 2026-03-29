Technical Overview: HealthComm (Current State)
1. The Patient Case Vector Manager (R4 Space)

Functionality:
The Vector Manager serves as the Data Persistence and Normalization Layer. It scales raw clinical inputs into a standardized 4-Dimensional Vector (R4) to ensure mathematical parity across disparate units (e.g., converting kilometers and hours into a unified 0–10 scale).

Vector Composition (V∈R4):

x0​: Urgency Score (Scaled 0–10).
  
x1​: Time Intensity (Scaled 0–10, max 24h).

x2​: Clinical Stability (Scaled 0–10, tracking improvement/decline).

x3​: Physical Distance (Scaled 0–10, max 100km).

Current Engineering Trade-offs:

  Dimensional Reduction: Unlike the proposed 8D model, the current system uses a 4D spatial vector paired with a Specialty Reference Anchor. This reduces computational overhead in low-connectivity environments while maintaining high-fidelity triage.    Reference Anchors vs. One-Hot Encoding: Instead of encoding the specialty into the vector, the system compares the patient vector against a static Specialty Reference Vector (e.g., Cardiology: [10,3,10,2]). This allows for "Directional Triage" based on how closely a patient’s distress matches a specific clinical profile.

2. The Clinical Logic Bridge

Concept:
The Bridge acts as a Structural Adapter between the PostgreSQL persistence layer and the BufferedPriorityQueue. It handles the "Object-Relational Mapping" of clinical cases into the Patient dataclass.

Mechanism:
The Bridge performs Invariant Validation, ensuring every vector processed is exactly 4D. It also handles the mapping of database specialty codes (integers) to human-readable strings for the logic engine.

Why the Bridge exists:

Encapsulation: It isolates the database logic (psycopg2) from the priority math (numpy).
Buffered Processing: It allows the system to load "Open" cases from the database into a high-performance Min-Heap for real-time sorting.
