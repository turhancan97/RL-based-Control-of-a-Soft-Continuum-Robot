'''
    Author: Turhan Can KARGIN
    Python Version: 3.9.7
    Visualization of the task space of the robot
'''
# %% import necessary libraries
import sys # to include the path of the package
sys.path.append('../')
import numpy as np
from kinematics.forward_velocity_kinematics import trans_mat_cc, coupletransformations
import matplotlib.pyplot as plt
plt.style.use('../continuum_robot/plot.mplstyle')
from continuum_robot.utils import *

# from configuration space (kappa, length) to task space (x,y)
# %% Section 1: With known kappa values
## In this section, the all curvature values will be same and change from -4 to 16 (limit for our robot).
## The aim is to see how the robot moves in the task space.
# parameters (kappa1, kappa2, kappa3, length1, length2, length3)
kappa = np.arange(-4,14.1,0.1) # 1/m
l = 0.1000 # m

plt.subplot(2, 2, 1)
# simulation for seeing task space
for kappa_val in kappa:
    T1_cc = trans_mat_cc(kappa_val+2,l)
    T1_tip = np.reshape(T1_cc[len(T1_cc)-1,:],(4,4),order='F');
    
    # section 2
    T2 = trans_mat_cc(kappa_val+1,l);
    T2_cc = coupletransformations(T2,T1_tip);
    T2_tip = np.reshape(T2_cc[len(T2_cc)-1,:],(4,4),order='F');
    
    # section 3
    T3 = trans_mat_cc(kappa_val,l);
    T3_cc = coupletransformations(T3,T2_tip);
    
    # Plot the trunk with three sections and point the section seperation
    #plt.scatter(T1_cc[0,12],T1_cc[0,13],linewidths=5,color = 'black')
    plt.plot([-0.025, 0.025],[0,0],'black',linewidth=5)
    plt.plot(T1_cc[:,12],T1_cc[:,13],'b',linewidth=3)
    #plt.scatter(T1_cc[-1,12],T1_cc[-1,13],linewidths=5,color = 'black')
    plt.plot(T2_cc[:,12],T2_cc[:,13],'r',linewidth=3)
    #plt.scatter(T2_cc[-1,12],T2_cc[-1,13],linewidths=5,color = 'black')
    plt.plot(T3_cc[:,12],T3_cc[:,13],'g',linewidth=3)
    plt.scatter(T3_cc[-1,12],T3_cc[-1,13],linewidths=5,color = 'black')

plt.grid(visible=True)
plt.title("Task Space of Planar Continuum Robot with Known Curvatures")
plt.xlabel("X - Position [m]")
plt.ylabel("Y - Position [m]")
plt.show()
# %% Section 2: With random kappa values
## In this section, the all curvature values will be uniformly random and beween -4 and 16 (limit for our robot).
## The aim is to see all posible curvature values in the task space so that we can have idea of the robot's behaviour.
## This will help us to max, min state in RL environment.

# parameters (kappa1, kappa2, kappa3, length1, length2, length3)
size = 10000 # make it bigger to get more accurate result
kappa1 = np.random.uniform(low=-4, high=16, size=(size,)) # 1/m
kappa2 = np.random.uniform(low=-4, high=16, size=(size,))
kappa3 = np.random.uniform(low=-4, high=16, size=(size,))
l = 0.1000 # m

x = []
y = []
plt.subplot(2, 2, 2)
for i in range(size):
    print(f'Sample: {i}', end='\r')
    T1_cc = trans_mat_cc(kappa1[i],l)
    T1_tip = np.reshape(T1_cc[len(T1_cc)-1,:],(4,4),order='F');
    
    # section 2
    T2 = trans_mat_cc(kappa2[i],l);
    T2_cc = coupletransformations(T2,T1_tip);
    T2_tip = np.reshape(T2_cc[len(T2_cc)-1,:],(4,4),order='F');
    
    # section 3
    T3 = trans_mat_cc(kappa3[i],l);
    T3_cc = coupletransformations(T3,T2_tip);
    
    # Plot the trunk with three sections and point the section seperation
    #plt.scatter(T1_cc[0,12],T1_cc[0,13],linewidths=5,color = 'black')
    
    plt.plot([-0.025, 0.025],[0,0],'black',linewidth=5)
    plt.plot(T1_cc[:,12],T1_cc[:,13],'b',linewidth=1,alpha=0.01)
    #plt.scatter(T1_cc[-1,12],T1_cc[-1,13],linewidths=5,color = 'black')
    plt.plot(T2_cc[:,12],T2_cc[:,13],'r',linewidth=1,alpha=0.01)
    #plt.scatter(T2_cc[-1,12],T2_cc[-1,13],linewidths=5,color = 'black')
    plt.plot(T3_cc[:,12],T3_cc[:,13],'g',linewidth=1,alpha=0.01)
    plt.scatter(T3_cc[-1,12],T3_cc[-1,13],linewidths=2,color = 'black')
    x.append(T3_cc[-1,12])
    y.append(T3_cc[-1,13])

# plt.grid(linestyle=':', linewidth=1.5)
# plt.title("Task Space of Planar Continuum Robot",fontsize=20,fontweight="bold")
plt.xlabel("Position x - [m]",fontsize=20)
plt.ylabel("Position y - [m]",fontsize=20)
plt.savefig('../docs/images/task_space.png')
plt.show()

# %% 
# To just see the task space without robot
plt.scatter(x,y,linewidths=2,color = 'black')
plt.grid(linestyle=':', linewidth=1.5)
plt.title("Task Space of Planar Continuum Robot",fontsize=30,fontweight="bold")
plt.xlabel("Position x - [m]",fontsize=30)
plt.ylabel("Position y - [m]",fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('../docs/images/task_space_2.png')
plt.show()
# %%  Section 3: Plotting the position of the robot for a given curvature

# Uncommment the below code to see the position of the robot for a given curvature
# kappa = np.array((-4.0, -4.0, 16.0)) # 1/m
# l = 0.1000 # m

# T1_cc = trans_mat_cc(kappa[0],l)
# T1_tip = np.reshape(T1_cc[len(T1_cc)-1,:],(4,4),order='F');

# # section 2
# T2 = trans_mat_cc(kappa[1],l);
# T2_cc = coupletransformations(T2,T1_tip);
# T2_tip = np.reshape(T2_cc[len(T2_cc)-1,:],(4,4),order='F');

# # section 3
# T3 = trans_mat_cc(kappa[2],l);
# T3_cc = coupletransformations(T3,T2_tip);

# # Plot the trunk with three sections and point the section seperation
# #plt.scatter(T1_cc[0,12],T1_cc[0,13],linewidths=5,color = 'black')
# plt.plot([-0.025, 0.025],[0,0],'black',linewidth=5)
# plt.plot(T1_cc[:,12],T1_cc[:,13],'b',linewidth=3)
# #plt.scatter(T1_cc[-1,12],T1_cc[-1,13],linewidths=5,color = 'black')
# plt.plot(T2_cc[:,12],T2_cc[:,13],'r',linewidth=3)
# #plt.scatter(T2_cc[-1,12],T2_cc[-1,13],linewidths=5,color = 'black')
# plt.plot(T3_cc[:,12],T3_cc[:,13],'g',linewidth=3)
# plt.scatter(T3_cc[-1,12],T3_cc[-1,13],linewidths=5,color = 'black')

# print(T3_cc[-1,12],T3_cc[-1,13])

# %% Plot the kappa values
plt.subplot(2, 2, 3)
plt.plot(range(len(kappa)),kappa, c = 'black', label = "3rd Section Curvature Values")
plt.plot(range(len(kappa)),kappa+1,c = 'red', label = "2nd Section Curvature Values")
plt.plot(range(len(kappa)),kappa+2,c ='blue', label = "1st Section Curvature Values")
plt.title("Known Curvature Values of Planar Continuum Robot")
plt.xlabel("th Sample")
plt.ylabel("Curvature Value")
plt.legend()

plt.subplot(2, 2, 4)
plt.boxplot([list(kappa1),list(kappa2),list(kappa3)])
# plt.scatter(range(len(kappa3)),kappa3, label = "3rd Section Curvature Values", c = "black",linewidth=0.01,alpha=0.25)
# plt.scatter(range(len(kappa2)),kappa2, label = "2nd Section Curvature Values",c = 'red',linewidth=0.01,alpha=0.25)
# plt.scatter(range(len(kappa1)),kappa1, label = "1st Section Curvature Values",c = 'blue',linewidth=0.01,alpha=0.25)
plt.title("Random Curvature Values of Planar Continuum Robot")
plt.xlabel("Number of Curvature - Total 30000 Sample")
plt.ylabel("Curvature Value")
# plt.legend()
plt.show()