# MIT-Transfer-Project-2026
A documentation of my journey to MIT


# HealthComm: Vector-Based Triage & Communication System

## Project Overview
HealthComm is a healthcare communication system with leverages Artificial Intelligence, aimed at bridging communication gap between medical personnel and solving issues in low-connectivity environments.
It will be linked to the hospital's system through a Middleware which will be handled mostly by AI. I'm doing this to ensure privacy and security of the hospital's data and make translation between HealthComm  and the hospital's system.
This repository describes my journey as I build this system using mathematical concepts I'm studying.

## The Mathematical Model (MATH 121; Algebra and Trigonometry & MATH 123; Vectors and Geometry)
I represented patient cases as a vector in 5 dimensions
(based on observations I made in Ghanaian hospitals and clinics) 
I worked with Claude to implement this logic into a Python-based vector manager, $\mathbb{R}^5$:
$$\vec{P} = (x_1, x_2, x_3, x_4, x_5)$$

### Vector Components:
* **$x_1$ (Urgency Score):** A Discrete scale (1-10).
* **$x_2$ (Specialty Code):** Categorical mapping for medical departments.
* **$x_3$ (Time Intensity):** Continuous variable representing care hours.
* **$x_4$ (Stability Trend):** Rate of change in physiological vitals(with a negative value representing improvement and positive values representing worsening condition)
* **$x_5$ (Proximity):** Euclidean distance (km) between patient and facility.

### Logic I applied:
1.  **Magnitude Analysis:* The "Weight"(intensity or urgency) of a case using the Euclidean Norm ($L_2$ norm): 
    $\|\vec{P}\| = \sqrt{x_1^2 + x_2^2 + x_3^2 + x_4^2 + x_5^2}$
2.  **Set Theory (MATH 121):** Set-membership logic is used to manage access control, to make sure that the set of "Authorized Personnel" for a patient is formed from the intersection of the "Assigned Team" and "Active Shift" sets.

## Technical Implementation
* **Language:** Python
* **Libraries:** NumPy (for Vector math), Psycopg2 (Database connectivity)
* **Architecture:** Object-Oriented Programming (OOP) utilizing Python Dataclasses for strict data typing.

## Why did I choose to use vectors?
Usually programming concepts are based on "If-Then" statements. But by using vectors, calculating **Cosine Similarity** between cases to identify clusters of health crises in real-time and automate resource allocation can be done with mathematical precision.

##Acknowledgements & Tools:
I developed this project as part of a 10-week intensive study of DCIT (Computer science) and MATH courses. I use Claude (Anthropic) and Google's Gemini as pair-programming partners to help implement the Python class structures and database connectivity, allowing me to focus the research on applying Vector Geometry and Set Theory to healthcare triage.

## What I plan to do: 
Build a workflow system that manages medical staff schedules, determines and alerts appropriate staff of patient emergencies based on patient dat and also manage appointments.

##Motivation:
Making mistakes and discovering new efficient and faster solutions to them is really exciting. It makes me understand that there's always a way to make a system better.
Most exciting of all is knowing that the stuff I learn from my books can actually be implemented to form the base of such a practical project.
Looking forwad to consistently improve and be able to solve real world challenges!
