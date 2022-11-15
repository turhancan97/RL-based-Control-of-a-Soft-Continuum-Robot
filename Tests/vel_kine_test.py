import sys
sys.path.append('../')

import numpy as np
from kinematics.forward_velocity_kinematics import three_section_planar_robot, jacobian_matrix

# parameters
delta_kappa = 0.1; 
l1 = 0.1000; # metre
l2 = 0.1000; # metre
l3 = 0.1000; # metre

kappa1 = [1, 1.2, 1.4, 1.6, 1.8]
kappa2 = [2, 2.3, 2.6, 2.9, 2.12]
kappa3 = [0.01, 0.02, 0.03, 0.04, 0.05]


l = [l1, l2, l3] # Each section's length

x_real = []
y_real = []

x_vel = []
y_vel = []


# J = jacobian_matrix(delta_kappa, kappa1, kappa2, kappa3, l)


# Önce farklı kappalarda pozisyonu bul sonra hızları bul pozisyonun türevini karşılaştır
for i in range(4):
    T3_cc = three_section_planar_robot(kappa1[i],kappa2[i],kappa3[i],l)
    
    print("Positions")
    print("x: ", T3_cc[0,3])
    print("----------------")
    print("y: ", T3_cc[1,3])
    print("----------------")
    
    x_real.append(T3_cc[0,3])
    y_real.append(T3_cc[1,3])
    
    J = jacobian_matrix(delta_kappa, kappa1[i], kappa2[i], kappa3[i], l)
    vel_kappa = np.array([0.2,0.3,0.01])
    
    vel_tip = J @ vel_kappa
    print("Velocity of the Tip: ", vel_tip)
    
    x_vel.append(vel_tip[0])
    y_vel.append(vel_tip[1])
    
print("------------------------")
for i in range(3):
    print("Velocity x: ")
    print(x_real[i+1]-x_real[i])
    print("------------------------")
    print("Velocity y: ")
    print(y_real[i+1]-y_real[i])
    print("------------------------")