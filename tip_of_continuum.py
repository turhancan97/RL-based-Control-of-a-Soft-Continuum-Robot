import sys
sys.path.append('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Functions')

import numpy as np
import matplotlib.pyplot as plt
from forward_velocity_kinematics import three_section_planar_robot


# from configuration space (kappa, length) to task space (x,y)

# parameters
kappa1 = 0.5; # 1/m
l1 = 0.1000; # metre
kappa2 = 0.5000; # 1/m
l2 = 0.1000; # metre
kappa3 = 0.5000; # 1/m
l3 = 0.1000; # metre

# kappa = [kappa1, kappa2, kappa3] # Each section's curvature
l = [l1, l2, l3] # Each section's length

    
# section 1
T3_cc = three_section_planar_robot(kappa1,kappa2,kappa3,l);

# Plot the trunk with three sections and point the section seperation
plt.scatter(T3_cc[0,3],T3_cc[1,3],linewidths=5,color = 'black',label="Tip Position of the Third Section")
plt.legend(loc="best")
plt.title("Tip point of 2D Continuum Robot")
plt.xlabel("X - Position")
plt.ylabel("Y - Position")
plt.show()
