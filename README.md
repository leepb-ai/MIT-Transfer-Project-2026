# MIT-Transfer-Project-2026
A documentation of my journey to MIT


# HealthComm: Vector-Based Triage & Communication System

## Project Overview
HealthComm is a healthcare communication system with leverages Artificial Intelligence, aimed at bridging communication gap between medical personnel and solving issues in low-connectivity environments.
The system uses a 5-dimensional vector, whose dimensions are measure of urgency to dtermine patient priority during clinical triage.

It will be linked to the hospital's system through a Middleware which will be handled mostly by AI. I'm doing this to ensure privacy and security of the hospital's data and make translation between HealthComm  and the hospital's system.
This repository describes my journey as I build this system using mathematical concepts I'm studying.

## The Mathematical Model (MATH 121; Algebra and Trigonometry & MATH 123; Vectors and Geometry)

## Technical Implementation
* **Language:** Python
* **Libraries:** NumPy (for Vector math), Psycopg2 (Database connectivity)
* **Architecture:** Object-Oriented Programming (OOP) utilizing Python Dataclasses for strict data typing.

## Why did I choose to use vectors?
Usually programming concepts are based on "If-Then" statements. But by using vectors, calculating **Cosine Similarity** between cases to identify clusters of health crises in real-time and automate resource allocation can be done with mathematical precision.

##Acknowledgements & Tools:
I'm developing this project as part of a study of DCIT (Computer science) and MATH courses. I use Claude (Anthropic) and Google's Gemini as pair-programming partners to help implement the Python class structures and database connectivity, allowing me to focus the research on applying Vector Geometry and Set Theory to healthcare triage.

## What I plan to do: 
Build a workflow system that determines and alerts appropriate staff of patient emergencies based on patient data.

🚀 Future Engineering: Beyond JSON & Text
To ensure HealthComm remains functional in the most restrictive network environments (Edge/Low-Bandwidth), the next phase of development focuses on two critical infrastructure upgrades:

1. Natural Language Processing (NLP) Layer
Currently, the system relies on structured inputs to build patient vectors. I am implementing an NLP Middleware that will parse unstructured clinical notes into our 5D vector space. This ensures that medical staff can input data naturally while the system maintains mathematical precision for triage.

2. Optimized Binary Transmission (Protobuf/Custom Bit-Packing)
Standard JSON is verbose and carries significant overhead, which is a bottleneck in low-connectivity areas. I am transitioning the data transmission protocol from standard string-based JSON to a Custom Binary Format.

Why? By packing data into bit-fields and using protocol buffers, we can reduce packet size by up to 80%.

Impact: This allows critical triage vectors to be transmitted over extremely weak signals (e.g., 2G or SMS-based protocols) where standard web requests would fail.


##Motivation:
Making mistakes and discovering new efficient and faster solutions to them is really exciting. It makes me understand that there's always a way to make a system better.
Most exciting of all is knowing that the stuff I learn from my books can actually be implemented to form the base of such a practical project.
Looking forwad to consistently improve and be able to solve real world challenges!


graph TD
    A[Unstructured Clinical Notes] -->|NLP Layer| B(Vector Mapping)
    B --> C{Priority Algorithm}
    C -->|High Urgency| D[Satellite/Binary Packet]
    C -->|Low Urgency| E[Local DB Storage]
    D -->|80% Compression| F[Medical Staff Alert]




