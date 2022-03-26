import numpy as np
import numpy.matlib as nummat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import style
from matplotlib.patches import Polygon
from trans_mat_prbm import trans_mat_prbm
from plot_tdcr_prbm import plot_tdcr_prbm


# Geometrical and mechanical properties of the 2 sections TDCR

# Subsegment number and length
n = np.array([10,10]).reshape(2, 1); #number of spacerdisks on the 2 sections, n[1] corresponds to proximal section
n_disk= np.sum(n); #number of spacerdisks
L = np.array([200e-3,200e-3]).reshape(2, 1); # Lenght of the section in m
l=np.array([L[0]/n[0]*np.ones(n[0],dtype=int),L[1]/n[1]*np.ones(n[1],dtype=int)]).reshape(1,20); #array of segment lengths of each subsegment

# Tendon position on the disks
r_disk=10*1e-3; #distance of routing channels to backbone-center[m]
# Tendons actuating the proximal segment
r1=np.array([r_disk,0,0,1]).reshape(4,1); 
r2=np.array([0,r_disk,0,1]).reshape(4,1); 
r3=np.array([-r_disk,0,0,1]).reshape(4,1); 
r4=np.array([0,-r_disk,0,1]).reshape(4,1); 
# Tendons actuating the distal segment
r5=np.array([r_disk*np.cos(np.pi/4),r_disk*np.sin(np.pi/4),0,1]).reshape(4,1); 
r6=np.array([r_disk*np.cos(3*np.pi/4),r_disk*np.sin(3*np.pi/4),0,1]).reshape(4,1); 
r7=np.array([r_disk*np.cos(5*np.pi/4),r_disk*np.sin(5*np.pi/4),0,1]).reshape(4,1); 
r8=np.array([r_disk*np.cos(-1*np.pi/4),r_disk*np.sin(-1*np.pi/4),0,1]).reshape(4,1); 
p_tendon=np.concatenate((r1,r2,r3,r4,r5,r6,r7,r8),axis=1); #additional tendons can be added through additional columns


# Tendons actuating the proximal segment
r1=np.array([0,r_disk,0,1]).reshape(4,1); 
r2=np.array([r_disk*np.cos(-1*np.pi/6),r_disk*np.sin(-1*np.pi/6),0,1]).reshape(4,1); 
r3=np.array([r_disk*np.cos(7*np.pi/6),r_disk*np.sin(7*np.pi/6),0,1]).reshape(4,1); 
# Tendons actuating the distal segment
r4=np.array([0,r_disk,0,1]).reshape(4,1); 
r5=np.array([r_disk*np.cos(-1*np.pi/6),r_disk*np.sin(-1*np.pi/6),0,1]).reshape(4,1); 
r6=np.array([r_disk*np.cos(7*np.pi/6),r_disk*np.sin(7*np.pi/6),0,1]).reshape(4,1); 
p_tendon=np.concatenate((r1,r2,r3,r4,r5,r6),axis=1); #additional tendons can be added through additional columns

# Tendon tension
# F = [8 2 8 2 0 0 0 0];
F = np.array([8,0,0,0,0,0]).reshape(1,6);

# Backbone mechanical and geometrical properties
E=54*10**9;# Youngs modulus
nu=0.3; #Poissons ratio
G=E/(2*(1+nu)); #Shear modulus
ro=1.4/2*10**(-1*3); #outer radius of bb
ri=0; #inner radius of bb
I=(1/4)*np.pi*((ro**4)-(ri**4)); #moment of inertia
g=0; #9.81; #acceleration due to gravity
m_bb=0.0115*l*g; #mass of backbone #weight of the backbone expressed in kg/m multiplied by length and g
m_disk=0.2*1e-3*np.ones(n_disk,dtype=int)*g; #array of masses of each spacerdisk

# External tip forces and moments

Ftex = np.array([0,0,0,0]).reshape(4,1); # Force applied at the tip, expressed in global frame
Mtex = np.array([0,0,0,0]).reshape(4,1); # Moment applied at the tip

# Number of rigid bodies
nrb = 4;
# Length of rigid bodies, optimized with particle swarm algorithm in Chen 2011
gamma= np.array([0.125,0.35,0.388,0.136]).reshape(1,4)/np.sum(np.array([0.136,0.388,0.35,0.125]).reshape(1,4));

#phi0 = pi/2;
phi0 = 0;

# Initialization
rep = np.concatenate((np.array([phi0]),0*np.ones(nrb,dtype=int)),axis=0)
var0 = nummat.repmat(rep,1,n_disk);