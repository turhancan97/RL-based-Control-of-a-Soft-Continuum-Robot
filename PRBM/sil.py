import numpy as np
import numpy.matlib as nummat

#function [T,Trb] = trans_mat_prbm(var,nrb,gamma,l,q,p)
def trans_mat_prbm(var,nrb,gamma,l,q,p):
    ##retruns rotation matrix from frame q to p
    T=np.eye(4);
    Trb = np.zeros((4,4,nrb*(q-p))); # trnasformation matrix from i-1 disk to Pj
    for iteration in range(p,q):
        theta = var[0][((nrb+1)*(iteration+1)-nrb-1):(nrb+1)*(iteration+1)-nrb+2];
        print(theta)
        print("--------------------------------------")
        phi = var[0][((nrb+1)*(iteration+1)-nrb+2)];
        epsi = var[0][((nrb+1)*(iteration+1)-nrb+3)];
            
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
        Trb[:,:,nrb*((iteration+1)-(p+1))] = np.matmul(T,Ti);
            
        for k in range(0,nrb-1):
            temp = np.array([np.cos(theta[k]),0,np.sin(theta[k]),
                          gamma[0][k+1]*l[0][iteration]*np.sin(theta[k]),
                          0,1,0,0,
                          -1*np.sin(theta[k]),
                          0,np.cos(theta[k]),gamma[0][k+1]*l[0][iteration]*np.cos(theta[k]),
                          0, 0, 0, 1]).reshape(4,4);
            Ti = np.matmul(Ti,temp);
            Trb[:,:,nrb*((iteration+1)-(p+1))+k+1] = np.matmul(T,Ti);
            
    T = np.matmul(T,Ti,R_phi_epsi);
    return T,Trb


######################
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
            Trb[:,:,nrb*((iteration+1)-(p+1))] = np.matmul(T,Ti);
            
            for k in range(0,nrb-1):
                temp = np.array([np.cos(theta[k]),0,np.sin(theta[k]),
                          gamma[0][k+1]*l[0][iteration]*np.sin(theta[k]),
                          0,1,0,0,
                          -1*np.sin(theta[k]),
                          0,np.cos(theta[k]),gamma[0][k+1]*l[0][iteration]*np.cos(theta[k]),
                          0, 0, 0, 1]).reshape(4,4);
                Ti = np.matmul(Ti,temp);
                Trb[:,:,nrb*((iteration+1)-(p+1))+k+1] = np.matmul(T,Ti);
                
            T = np.matmul(T,Ti,R_phi_epsi);
            return T,Trb


###############

var = np.arange(0,100).reshape(1,100)
nrb = 4
gamma = np.array([[0.12512513, 0.35035035, 0.38838839, 0.13613614]])
l = np.array([[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]]).reshape(1,20)
q = 10
p = 0 
T,Trb = trans_mat_prbm(var,nrb,gamma,l,q,p) 


