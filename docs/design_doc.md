docs: initialize design documentation

The idea behind patient_case_vecor_manager.py;
It handles patient prioritization using a 5-dimensional vector space, and this idea stemmed from the concept vectors.

By representing a variable patient_case as a vector, the various cases which would indicate/'propose' a patient's case is to be deemed urgent and the it's level of urgency are assigned dimensions of the patient_case_vector, in this case 5 dimensions(x1-urgency score,x2-specialty code,x3-time intensity,x4-stability trend,x5-distance).

By definition:

x1: has an integer scale 1-10 
#the scale here allows the urgency to provide a sizeable range to allow levels of urgency to be properly quantified.

x2: it is an integer representation of different specialties(1-cardiology,2-pediatrics,3-ER,4-dental)
#by assigning the specialty codes to specific integers, severity can be computed as actual figures

x3: simply the hours of care required 
#the greater the value, the more severe the case

x4: this represents the rate of change: where a negative value depicts improving condition and a positive value worsening condition
#this is calculated from the rate of change of vitals in a certain time

x5: it represents the distance in kilometers from the hospital

The urgency is calculated then by finding the magnitude of the 5-dimensional vector(based on the Euclidean Norm)


#Thoughts
I do realise that there is an issue concerning the accuracy of the urgency.
Especially with regards to x2(Specialty code); per the logic behind it, the system considers ER for instance, which is assigned an integer of 3 to be greater, in terms of importance, than cardiology.
There are also other factors that need refining.
Which is not valid..I'm going to think of a way to correct that error soon.



