
#function [p_f] = plot_tdcr_prbm(nrb,gamma,l,p_tendon,n_disk,var,r_disk)

## visualisation of the results
n_tendon = p_tendon.shape[1];
p_last=np.array([0,0,0]).reshape(3,1);

#[T_i,Trb]=trans_mat_prbm(var,nrb,gamma,l,n_disk,0);
p_plot = p_last;
Trb = np.ones((4,4,80))
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
ax = fig.add_subplot(111, projection='3d')
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
fig = plt.figure()
ax = fig.gca(projection='3d')
#plotting tendons
ax.plot(np.concatenate([p_t1[0,:].reshape(1,6),p_t2[0,:].reshape(1,6)],axis=0).flatten(),np.concatenate([p_t1[1,:].reshape(1,6),p_t2[1,:].reshape(1,6)],axis=0).flatten(),np.concatenate([p_t1[2,:].reshape(1,6),p_t2[2,:].reshape(1,6)],axis=0).flatten())
#plot3([p_t1(1,:);p_t2(1,:)],[p_t1(2,:);p_t2(2,:)],[p_t1(3,:);p_t2(3,:)],'r','LineWidth',1)
plt.show()