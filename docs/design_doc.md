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


Great News!
After hours of research and thinking(I barely slept), I was finally able to find some method to overcome my initial patient_case_vector_manager:
1. I used the one-Hot Encoding to remove the disadvantage of using integers to represent the specialty code, and I did this because the One-Hot Encoding allows me to represent each code with a binary vector which assigns a '1' based on that particular code's index. Say ER is the fourth index, it is represented by[ 0, 0, 0, 1 ].
Here the system does not regard Er as greater than say Cardiology, since they are stored based on their index or positions.
2. I also found an interesting formula, The Min-Max formula, and it beautifully brigdes the gap of scalability between the dimensions. What the Min-Max formula does is to assign the dimenions to a minimum and maximum limits [0,1](note my use of closed sets her indicating that the values will be within the range of 0 and 1 inclusive).
admire this formula because it reduces all the values of the dimensions to [0,1], thus regardless of how large a value from a given dimension is, it'll still be treated equally as the rest as they are all reduced to fall within the same range.(It is implemented in the code by the Vector scalar function)
And it noticeably corresponds to the Speciality code dimension as it also falls in the [0,1] range
This allows tthe Euclidean Distance to be calculated accurately.

#Side-note;
Despite the fact that this seems brilliant, per my point of view I can still tell there are more ways to improve on this and build a more effiient system.

