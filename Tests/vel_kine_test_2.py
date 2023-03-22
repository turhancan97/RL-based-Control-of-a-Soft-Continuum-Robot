import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from kinematics.forward_velocity_kinematics import three_section_planar_robot, jacobian_matrix

# parameters
delta_kappa = 0.1; 
kappa1 = 2.0; # 1/m
l1 = 0.1000; # metre
kappa2 = 1.5000; # 1/m
l2 = 0.1000; # metre
kappa3 = 1.3000; # 1/m
l3 = 0.1000; # metre

l = [l1, l2, l3] # Each section's length

#J = jacobian_matrix(delta_kappa, kappa1, kappa2, kappa3, l)

t_matrix = three_section_planar_robot(kappa1,kappa2,kappa3,l)
#quat = quaternion.as_float_array(quaternion.from_rotation_matrix(t_matrix))
tip_point = np.array([t_matrix[0,3],t_matrix[1,3]])
J = jacobian_matrix(delta_kappa, kappa1, kappa2, kappa3, l)
vel_kappa = np.array([0.001,0.1,1])



total_time = 5  # (seconds)
dt = 0.1 # sample sizes
time_stamp = int(total_time/dt)
i = 0


x_pos = np.zeros((2, time_stamp))
x_vel = np.zeros((2, time_stamp))


while i <= time_stamp-1:
    x_pos[:,i] = tip_point
    x_vel[:,i] = J @ vel_kappa
    tip_point += x_vel[:,i] * dt
    plt.scatter(x_pos[0][i],x_pos[1][i],linewidths=2.5,color = 'blue')
    plt.pause(0.005)
    kappa1 += 0.0001
    kappa2 += 0.01
    kappa3 += 0.1
    J = jacobian_matrix(delta_kappa, kappa1, kappa2, kappa3, l)
    i += 1
    print(x_pos)
    #time.sleep(0.1)


kappa1 = 2.0; # 1/m
l1 = 0.1000; # metre
kappa2 = 1.5000; # 1/m
l2 = 0.1000; # metre
kappa3 = 1.3000; # 1/m
l3 = 0.1000; # metre

l = [l1, l2, l3] # Each section's length

for j in range(6):
    t_matrix = three_section_planar_robot(kappa1,kappa2,kappa3,l)
    tip_point = np.array([t_matrix[0,3],t_matrix[1,3]])
    plt.scatter(tip_point[0],tip_point[1],linewidths=5,color = 'black')
    plt.pause(0.05)
    kappa1 = kappa1 + 0.001; # 1/m
    kappa2 = kappa2 + 0.1; # 1/m
    kappa3 = kappa3 + 1; # 1/m

plt.scatter(x_pos[0][i-1],x_pos[1][i-1],linewidths=2.5,color = 'blue',label = "Velocity Kinematics Motion")
plt.scatter(tip_point[0],tip_point[1],linewidths=5,color = 'black',label="Actual Motion")
plt.legend(loc="best")
plt.title("2D Motion of Tip of the Continuum Robot")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.show()