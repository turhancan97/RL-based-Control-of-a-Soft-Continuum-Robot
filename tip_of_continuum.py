import numpy as np
import matplotlib.pyplot as plt

# Planar Robot Kinematics Functions

# Homogeneous transformation matrix
def three_section_planar_robot(kappa, l):
#Mapping from configuration parameters to task space for the tip of the continuum robot
#
# INPUT:
# kappa: curvature
# l: trunk length
#
# OUTPUT:
# T: Transformation matrices containing orientation and position
    
    c_ks = np.cos((kappa[0]*l[0])+(kappa[1]*l[1])+(kappa[2]*l[2]));
    
    s_ks = np.sin((kappa[0]*l[0])+(kappa[1]*l[1])+(kappa[2]*l[2]));
    
    A_14 = ((np.cos(kappa[0]*l[0])-1)/kappa[0]) + ((np.cos((kappa[0]*l[0])+(kappa[1]*l[1]))-np.cos(kappa[0]*l[0]))/kappa[1]) + ((np.cos((kappa[0]*l[0])+(kappa[1]*l[1])+(kappa[2]*l[2]))-np.cos((kappa[0]*l[0])+(kappa[1]*l[1])))/kappa[2])
    
    A_24 = ((np.sin(kappa[0]*l[0]))/kappa[0]) + ((np.sin((kappa[0]*l[0])+(kappa[1]*l[1]))-np.sin(kappa[0]*l[0]))/kappa[1]) + ((np.sin((kappa[0]*l[0])+(kappa[1]*l[1])+(kappa[2]*l[2]))-np.sin((kappa[0]*l[0])+(kappa[1]*l[1])))/kappa[2])

    T = np.array([c_ks,s_ks,0,0,-s_ks,c_ks,0,0,0,0,1,0,A_14,A_24,0,1]);
        
    T = np.reshape(T,(4,4),order='F');
    return T



# from configuration space (kappa, length) to task space (x,y)

# parameters
kappa1 = 1.7035; # 1/m
l1 = 0.1000; # metre
kappa2 = 1.0000; # 1/m
l2 = 0.1000; # metre
kappa3 = 2.0000; # 1/m
l3 = 0.1000; # metre

kappa = [kappa1, kappa2, kappa3] # Each section's curvature
l = [l1, l2, l3] # Each section's length

    
# section 1
T3_cc = three_section_planar_robot(kappa,l);

# Plot the trunk with three sections and point the section seperation
plt.scatter(T3_cc[0,3],T3_cc[1,3],linewidths=5,color = 'black',label="Tip of the Third Section")
plt.legend(loc="best")
plt.title("Tip of 2D Continuum Robot")
plt.xlabel("X - Position")
plt.ylabel("Y - Position")
plt.show()
