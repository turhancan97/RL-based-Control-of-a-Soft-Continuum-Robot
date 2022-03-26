import numpy as np
import numpy.matlib as nummat
#%matplotlib widget
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import style
from matplotlib.patches import Polygon
from trans_mat_prbm import trans_mat_prbm

def plot_tdcr_prbm(nrb,gamma,l,p_tendon,n_disk,var,r_disk):
    n_tendon = p_tendon.shape[1];
    p_last=np.array([0,0,0]).reshape(3,1);
    
    [T_i,Trb]=trans_mat_prbm(var,nrb,gamma,l,n_disk,0);
    p_plot = p_last;
    for j in range(0,Trb.shape[0]):
        p_plot = np.concatenate((p_plot,Trb[j,0:3,3].reshape(3,1)),axis=1);
    
    p_f = p_last;
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    lmax=np.sum(l)
    ax.set_xlim(-lmax,lmax)
    ax.set_ylim(-lmax,lmax)
    ax.set_zlim(-lmax*(2/3),(4/3)*lmax)
    x = [0.07,0.07,-0.07,-0.07]
    y = [-0.07,0.07,0.07,-0.07]
    z = [0,0,0,0]
    verts = [list(zip(x,y,z))]
    ax.add_collection3d(Poly3DCollection(verts,alpha=0.65,facecolor='#000000'))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    
    for ss_i in range(1,n_disk+1):
        
        # Tendon initial and final position
        T_i_i=trans_mat_prbm(var,nrb,gamma,l,ss_i-1,0); #returns transformation matrix from i to i-1
        T_i = T_i_i[0]
        p_t1=np.matmul(T_i,p_tendon); # coordinates of tends on disk ss_i-1
        T_i1_i1=trans_mat_prbm(var,nrb,gamma,l,ss_i,0);
        T_i1 = T_i1_i1[0]
        p_t2=np.matmul(T_i1,p_tendon);
            
        ## Plot 3D
        ax.plot(p_plot[0,:],p_plot[1,:],p_plot[2,:],'black') #plotting the backbone
        p_last=p_plot[:,nrb*ss_i].reshape(3,1);
        p_f = np.concatenate((p_f,p_plot),axis=1);
            
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
        x1= p_disk[0,:]
        y1= p_disk[1,:]
        z1= p_disk[2,:]
        verts_2 = [list(zip(x1, y1, z1))]
        srf = Poly3DCollection(verts_2, alpha=.6, facecolor='C0')
        plt.gca().add_collection3d(srf)
        #ax.add_patch(Polygon(xy, fill=True, color='palegreen'))
        #patch(p_disk(1,:)',p_disk(2,:)',p_disk(3,:)',[0 0.8 0.8]);
        
        #plotting tendons
        #ax.plot(np.concatenate([p_t1[0,:].reshape(1,6),p_t2[0,:].reshape(1,6)],axis=0),np.concatenate([p_t1[1,:].reshape(1,6),p_t2[1,:].reshape(1,6)],axis=0),np.concatenate([p_t1[2,:].reshape(1,6),p_t2[2,:].reshape(1,6)],axis=0))
        a = np.array([p_t1[0,:][5],p_t2[0,:][2]])
        b = np.array([p_t1[1,:][5],p_t2[1,:][2]])
        c = np.array([p_t1[2,:][5],p_t2[2,:][2]])
        ax.plot(a,b,c,'red')
        d = np.array([p_t1[0,:][0],p_t2[0,:][0]])
        e = np.array([p_t1[1,:][0],p_t2[1,:][0]])
        f = np.array([p_t1[2,:][0],p_t2[2,:][0]])
        ax.plot(d,e,f,'red')
        g = np.array([p_t1[0,:][4],p_t2[0,:][1]])
        h = np.array([p_t1[1,:][4],p_t2[1,:][1]])
        i = np.array([p_t1[2,:][4],p_t2[2,:][1]])
        ax.plot(g,h,i,'red')
    plt.show()
    return p_f