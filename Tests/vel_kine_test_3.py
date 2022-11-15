import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from kinematics.forward_velocity_kinematics import three_section_planar_robot, jacobian_matrix


# parameters
delta_kappa = 0.1; 
kappa1 = 1.0; # 1/m
l1 = 0.1000; # metre
kappa2 = 1.0000; # 1/m
l2 = 0.1000; # metre
kappa3 = 1.0000; # 1/m
l3 = 0.1000; # metre

l = [l1, l2, l3] # Each section's length

#J = jacobian_matrix(delta_kappa, kappa1, kappa2, kappa3, l)

t_matrix = three_section_planar_robot(kappa1,kappa2,kappa3,l)
#quat = quaternion.as_float_array(quaternion.from_rotation_matrix(t_matrix))
tip_point = np.array([t_matrix[0,3],t_matrix[1,3]])
J = jacobian_matrix(delta_kappa, kappa1, kappa2, kappa3, l)
vel_kappa = np.array([[0.05,0.05,0.05],[0.09,0.4,0.08],[1.0,1.25,1.5]])

total_time = 20  # (seconds)
dt = 0.1 # sample sizes
time_stamp = int(total_time/dt)
i = 0


x_pos = np.zeros((2, time_stamp*len(vel_kappa)))
x_vel = np.zeros((2, time_stamp*len(vel_kappa)))



for j in range(len(vel_kappa)):
    vel_kappa_new = vel_kappa[j,:]
    while i <= time_stamp-1:
        x_pos[:, time_stamp*j + i] = tip_point
        x_vel[:, time_stamp*j + i] = J @ vel_kappa_new
        tip_point += x_vel[:, time_stamp*j + i] * dt
        plt.scatter(x_pos[0][time_stamp*j + i],x_pos[1][time_stamp*j + i])
        plt.pause(0.05)
        kappa1 += vel_kappa_new[0] * dt
        kappa2 += vel_kappa_new[1] * dt
        kappa3 += vel_kappa_new[2] * dt
        J = jacobian_matrix(delta_kappa, kappa1, kappa2, kappa3, l)
        i += 1
        print(x_pos)
        #time.sleep(0.1)
    i = 0

# plt.show()

kappa1 = 1.0; # 1/m
l1 = 0.1000; # metre
kappa2 = 1.0000; # 1/m
l2 = 0.1000; # metre
kappa3 = 1.0000; # 1/m
l3 = 0.1000; # metre

l = [l1, l2, l3] # Each section's length

for j in range(len(vel_kappa)):
    vel_kappa_new = vel_kappa[j,:]
    for k in range(21):
        t_matrix = three_section_planar_robot(kappa1,kappa2,kappa3,l)
        tip_point = np.array([t_matrix[0,3],t_matrix[1,3]])
        plt.scatter(tip_point[0],tip_point[1],linewidths=5,color = 'black')
        plt.pause(0.05)
        kappa1 = kappa1 + vel_kappa_new[0]; # 1/m
        kappa2 = kappa2 + vel_kappa_new[1]; # 1/m
        kappa3 = kappa3 + vel_kappa_new[2]; # 1/m
    
plt.show()

# kappa1 = 1.7035+vel_kappa[0]; # 1/m
# l1 = 0.1000; # metre
# kappa2 = 1.0000+vel_kappa[1]; # 1/m
# l2 = 0.1000; # metre
# kappa3 = 2.0000+vel_kappa[2]; # 1/m
# l3 = 0.1000; # metre

# l = [l1, l2, l3] # Each section's length

# t_matrix2 = three_section_planar_robot(kappa1,kappa2,kappa3,l)
# tip_point2 = np.array([t_matrix2[0,3],t_matrix2[1,3]])

# print(tip_point)
# print(tip_point2)