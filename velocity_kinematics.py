import numpy as np
import matplotlib.pyplot as plt


# Homogeneous transformation matrix
def three_section_planar_robot(kappa1, kappa2, kappa3, l):
#Mapping from configuration parameters to task space for the tip of the continuum robot
#
# INPUT:
# kappa: curvature
# l: trunk length
#
# OUTPUT:
# T: Transformation matrices containing orientation and position
    
    c_ks = np.cos((kappa1*l[0])+(kappa2*l[1])+(kappa3*l[2]));
    
    s_ks = np.sin((kappa1*l[0])+(kappa2*l[1])+(kappa3*l[2]));
    
    A_14 = ((np.cos(kappa1*l[0])-1)/kappa1) + ((np.cos((kappa1*l[0])+(kappa2*l[1]))-np.cos(kappa1*l[0]))/kappa2) + ((np.cos((kappa1*l[0])+(kappa2*l[1])+(kappa3*l[2]))-np.cos((kappa1*l[0])+(kappa2*l[1])))/kappa3)
    
    A_24 = ((np.sin(kappa1*l[0]))/kappa1) + ((np.sin((kappa1*l[0])+(kappa2*l[1]))-np.sin(kappa1*l[0]))/kappa2) + ((np.sin((kappa1*l[0])+(kappa2*l[1])+(kappa3*l[2]))-np.sin((kappa1*l[0])+(kappa2*l[1])))/kappa3)

    T = np.array([c_ks,s_ks,0,0,-s_ks,c_ks,0,0,0,0,1,0,A_14,A_24,0,1]);
        
    T = np.reshape(T,(4,4),order='F');
    return T

def jacobian_matrix(delta_kappa, kappa1, kappa2, kappa3, l):
    
    J11 = (three_section_planar_robot(kappa1+delta_kappa,kappa2,kappa3,l)[0,3] - three_section_planar_robot(kappa1-delta_kappa,kappa2,kappa3,l))[0,3] / (2*delta_kappa);
    J12 = (three_section_planar_robot(kappa1,kappa2+delta_kappa,kappa3,l)[0,3] - three_section_planar_robot(kappa1,kappa2-delta_kappa,kappa3,l))[0,3] / (2*delta_kappa);
    J13 = (three_section_planar_robot(kappa1,kappa2,kappa3+delta_kappa,l)[0,3] - three_section_planar_robot(kappa1,kappa2,kappa3-delta_kappa,l))[0,3] / (2*delta_kappa);
    J21 = (three_section_planar_robot(kappa1+delta_kappa,kappa2,kappa3,l)[1,3] - three_section_planar_robot(kappa1-delta_kappa,kappa2,kappa3,l))[1,3] / (2*delta_kappa);
    J22 = (three_section_planar_robot(kappa1,kappa2+delta_kappa,kappa3,l)[1,3] - three_section_planar_robot(kappa1,kappa2-delta_kappa,kappa3,l))[1,3] / (2*delta_kappa);
    J23 = (three_section_planar_robot(kappa1,kappa2,kappa3+delta_kappa,l)[1,3] - three_section_planar_robot(kappa1,kappa2,kappa3-delta_kappa,l))[1,3] / (2*delta_kappa);
    
    J = np.array([J11,J12,J13,J21,J22,J23]);
    J = np.reshape(J,(2,3))
    
    return J



# parameters
delta_kappa = 0.1; 
kappa1 = 1.7035; # 1/m
l1 = 0.1000; # metre
kappa2 = 1.0000; # 1/m
l2 = 0.1000; # metre
kappa3 = 2.0000; # 1/m
l3 = 0.1000; # metre

l = [l1, l2, l3] # Each section's length

J = jacobian_matrix(delta_kappa, kappa1, kappa2, kappa3, l)

# # parameters
# kappa1 = 1.3; # 1/m
# l1 = 0.1000; # metre

# kappa2 = 1.1; # 1/m
# l2 = 0.1000; # metre

# kappa3 = 1.2; # 1/m
# l3 = 0.1000; # metre

 
# T1_cc = trans_mat_cc(kappa1,l1);

# x1 = T1_cc[0,3]
# y1 = T1_cc[1,3]

# T2_cc = trans_mat_cc(kappa2,l2);

# x2 = T2_cc[0,3]
# y2 = T2_cc[1,3]

# derx1 = (x1-x2)/0.2
# dery1 = (y1-y2)/0.2


# # By derrivative

# T1_cc_der = trans_mat_cc_der(kappa3,l3)


# derx2 = T1_cc_der[0,3]
# dery2 = T1_cc_der[1,3]



