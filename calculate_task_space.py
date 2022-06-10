import sys
sys.path.append('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Functions')

import numpy as np
import matplotlib.pyplot as plt
from forward_velocity_kinematics import trans_mat_cc, coupletransformations

# from configuration space (kappa, length) to task space (x,y)

# %% With known kappa values
# parameters
kappa = np.arange(-4,17,0.1) # 1/m
l = 0.1000 # m
    
plt.subplot(1, 2, 1)
# simulation for seeing task space
for kappa_val in kappa:
    T1_cc = trans_mat_cc(kappa_val,l)
    T1_tip = np.reshape(T1_cc[len(T1_cc)-1,:],(4,4),order='F');
    
    # section 2
    T2 = trans_mat_cc(kappa_val,l);
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
plt.title("Task Space of Planar Continuum Robot with Known Kappas")
plt.xlabel("X - Position [m]")
plt.ylabel("Y - Position [m]")

# %% With random kappa values
# parameters
size = 500 # make it bigger to get more accurate result
kappa1 = np.random.uniform(low=-4, high=16, size=(size,)) # 1/m
kappa2 = np.random.uniform(low=-4, high=16, size=(size,))
kappa3 = np.random.uniform(low=-4, high=16, size=(size,))
l = 0.1000 # m

plt.subplot(1, 2, 2)
for i in range(size):
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
    plt.plot(T1_cc[:,12],T1_cc[:,13],'b',linewidth=3)
    #plt.scatter(T1_cc[-1,12],T1_cc[-1,13],linewidths=5,color = 'black')
    plt.plot(T2_cc[:,12],T2_cc[:,13],'r',linewidth=3)
    #plt.scatter(T2_cc[-1,12],T2_cc[-1,13],linewidths=5,color = 'black')
    plt.plot(T3_cc[:,12],T3_cc[:,13],'g',linewidth=3)
    plt.scatter(T3_cc[-1,12],T3_cc[-1,13],linewidths=5,color = 'black')

plt.grid(visible=True)
plt.title("Task Space of Planar Continuum Robot with Random Kappas")
plt.xlabel("X - Position [m]")
plt.ylabel("Y - Position [m]")
plt.show()

# %% 

# kappa = np.array((6.750524852275853, 6.0441568774543715, 1.6081940656900415)) # 1/m
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