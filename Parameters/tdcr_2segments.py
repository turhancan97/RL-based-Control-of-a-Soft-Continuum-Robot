import numpy as np
import numpy.matlib as nummat
%matplotlib widget
# importing required libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon
from scipy.optimize import fsolve

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

###############################################################################
#function [T,Trb] = trans_mat_prbm(var,nrb,gamma,l,q,p)
def trans_mat_prbm(var,nrb,gamma,l,q,p):
    ##retruns rotation matrix from frame q to p
    global T, Trb
    T=np.eye(4);
    Trb = np.zeros((4,4,nrb*(q-p))); # trnasformation matrix from i-1 disk to Pj

    if q<p:
        print('Some error in rotation matrix indices')
    else:
        for iteration in range(p,q):
            theta = var[0][((nrb+1)*(iteration+1)-nrb):(nrb+1)*iteration-nrb+2];
            phi = var[0][((nrb+1)*(iteration+1)-nrb+3)];
            epsi = var[0][((nrb+1)*(iteration+1)-nrb+4)];
            
            R_phi = np.array([np.cos(phi),-1*np.sin(phi),0,0,
                              np.sin(phi),np.cos(phi),0,0,
                              0,0,1,0,
                              0,0,0,1]).reshape(4,4);
            
            R_phi_epsi = np.array([np.cos(epsi-phi),-1*np.sin(epsi-phi),0,0,
                          np.sin(epsi-phi),np.cos(epsi-phi),0,0
                          ,0,0,1,0,
                          0,0,0,1]).reshape(4,4);
            
            gamma_fun = np.array([1,0,0,0,0,1,0,0,0,0,1,gamma[0][0]*l[0][iteration],0,0,0,1]).reshape(4,4);
            
            Ti = np.matmul(R_phi,gamma_fun);
            Trb[:,:,nrb*(iteration-(p+1))] = np.matmul(T,Ti);
            
            for k in range(0,nrb-1):
                temp = np.array([np.cos(theta[k]),0,np.sin(theta[k]),
                          gamma[0][k+1]*l[0][iteration]*np.sin(theta[k]),
                          0,1,0,0,
                          -1*np.sin(theta[k]),
                          0,np.cos(theta[k]),gamma[0][k+1]*l[0][iteration]*np.cos(theta[k]),
                          0, 0, 0, 1]).reshape(4,4);
                Ti = np.matmul(Ti,temp);
                Trb[:,:,nrb*(iteration-(p+1))+k+1] = np.matmul(T,Ti);
                
            T = np.matmul(T,Ti,R_phi_epsi);
            return T,Trb











### Visual
## visualisation of the results
n_tendon = p_tendon.shape[1];
p_last=np.array([0,0,0]).reshape(3,1);

#sil
var = np.arange(0,100).reshape(1,100)

[T_i,Trb]=trans_mat_prbm(var,nrb,gamma,l,n_disk,0);
p_plot = p_last;
for j in range(0,Trb.shape[2]):
    p_plot = np.concatenate([p_plot,Trb[0:3,3,j].reshape(3,1)],axis=1);

p_f = p_last;

for ss_i in range(0,n_disk-18):
    
    # Tendon initial and final position
    #T_i=trans_mat_prbm(var,nrb,gamma,l,ss_i-1,0); #returns transformation matrix from i to i-1
    if ss_i == 0:
        T_i = np.eye(4)
        p_t1=np.matmul(T_i,p_tendon); # coordinates of tends on disk ss_i-1
        T_i1 = np.array([[0.9998,-0.0003,0.0173,0.0002],
                         [-0.0003  ,  0.9996   , 0.0296 ,   0.0003],
                         [-0.0173  , -0.0296  ,  0.9994 ,   0.0200],
                         [0      ,   0     ,    0   , 1.0000]])
        p_t2=np.matmul(T_i1,p_tendon);
    else:
        T_i = np.array([[0.9998 ,  -0.0003  ,  0.0173  ,  0.0002],
                        [-0.0003  ,  0.9996  ,  0.0296   , 0.0003],
                        [-0.0173 ,  -0.0296   , 0.9994   , 0.0200],
                        [0     ,    0      ,   0  ,  1.0000]])
        p_t1=np.matmul(T_i,p_tendon); # coordinates of tends on disk ss_i-1
        T_i1 = np.array([[0.9994  , -0.0010   , 0.0346  ,  0.0007],
                         [-0.0010  ,  0.9982  ,  0.0592  ,  0.0012],
                         [-0.0346  , -0.0592   , 0.9976   , 0.0400],
                         [0    ,     0     ,    0 ,   1.0000]])
        #T_i1=trans_mat_prbm(var,nrb,gamma,l,ss_i,0);
        p_t2=np.matmul(T_i1,p_tendon);
        
        
## Plot 3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(p_plot[0,:],p_plot[1,:],p_plot[2,:]) #plotting the backbone
lmax=np.sum(l)
ax.set_xlim(-lmax,lmax)
ax.set_ylim(-lmax,lmax)
ax.set_zlim(-lmax*(2/3),(4/3)*lmax)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
    # hold on;
    #     color = [1 1 1]*0.9;
    #     squaresize = 0.07;
    #     lmax=sum(l);
    #     axis([-lmax lmax -lmax lmax -lmax*2/3 4/3*lmax])
    #     fill3([1 1 -1 -1]*squaresize,[-1 1 1 -1]*squaresize,[0 0 0 0],color);
    #     xlabel('x (m)')
    #     ylabel('y (m)')
    #     zlabel('z (m)')
    #     grid on
    #     view([1 1 1])
    #     daspect([1 1 1])  
p_last=p_plot[:,nrb*ss_i].reshape(3,1);
p_f = np.concatenate([p_f,p_plot],axis=1);
    
    #plotting disks
t_disk=np.arange(0,2*np.pi,0.01).reshape(1,629); #parameter for plotting disk i
q = (r_disk+0.002)*T_i1[0:3,0:3];
w = np.concatenate((np.cos(t_disk),np.sin(t_disk),np.zeros(t_disk.shape)),axis=0);
t = np.matmul(q,w)
p_disk=nummat.repmat(p_last,1,t_disk.size)+t;
 
    ## Patch eklemeye bak
    #poly = Polygon(np.column_stack([p_disk[0,:].reshape(629,1),p_disk[1,:].reshape(629,1),p_disk[2,:].reshape(629,1)]), animated=True)
    #fig, ax = plt.subplots()
    #ax.add_patch(poly)
    #xy = [p_disk[0,:].reshape(629,1),p_disk[1,:].reshape(629,1),p_disk[2,:].reshape(629,1)]
    #ax.add_patch(Polygon(xy, fill=True, color='palegreen'))
    #patch(p_disk(1,:)',p_disk(2,:)',p_disk(3,:)',[0 0.8 0.8]);
#plotting tendons
#ax.plot(np.concatenate([p_t1[0,:].reshape(1,6),p_t2[0,:].reshape(1,6)],axis=0),np.concatenate([p_t1[1,:].reshape(1,6),p_t2[1,:].reshape(1,6)],axis=0),np.concatenate([p_t1[2,:].reshape(1,6),p_t2[2,:].reshape(1,6)],axis=0))
a = np.array([p_t1[0,:][5],p_t2[0,:][2]])
b = np.array([p_t1[1,:][5],p_t2[1,:][2]])
c = np.array([p_t1[2,:][5],p_t2[2,:][2]])
ax.plot(a,b,c)
d = np.array([p_t1[0,:][0],p_t2[0,:][0]])
e = np.array([p_t1[1,:][0],p_t2[1,:][0]])
f = np.array([p_t1[2,:][0],p_t2[2,:][0]])
ax.plot(d,e,g)
g = np.array([p_t1[0,:][4],p_t2[0,:][1]])
h = np.array([p_t1[1,:][4],p_t2[1,:][1]])
i = np.array([p_t1[2,:][4],p_t2[2,:][1]])
ax.plot(g,h,i)

#plot3([p_t1(1,:);p_t2(1,:)],[p_t1(2,:);p_t2(2,:)],[p_t1(3,:);p_t2(3,:)],'r','LineWidth',1)
plt.show()