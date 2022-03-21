# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:15:36 2022

@author: Asus
"""

## deneme

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
l=np.array([L[0]/n[0]*np.ones(n[0],dtype=int),L[1]/n[1]*np.ones(n[1],dtype=int)]); #array of segment lengths of each subsegment

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

#var,infodict = fsolve(optim_f,var0);
var = np.zeros(100)
## solver

    #function [res] = optim_f(var)
def optim_f(var):
    res=np.zeros((n_disk*(nrb+1),1),dtype=int); #nrb-1 revolute joints, 1 bending plane angle and 1 torsion angle
    F_prev = np.zeros((3,1),dtype=int);
    M_prev = np.zeros((3,1),dtype=int);
        for ss_i in range(n_disk,0,-1) #iterating over each subsegment
            
            # Kinematics
            T_i,Trb = trans_mat_prbm(var,nrb,gamma,l,ss_i,ss_i-1); #returns transformation matrix from i to i-1
            theta=var[list(range((nrb+1)*ss_i-nrb-1,(nrb+1)*ss_i-nrb+2,1))];
            phi = var[(nrb+1)*ss_i-nrb+2];
            ni = np.array([np.cos(phi+np.pi/2),np.sin(phi+np.pi/2),0]).reshape(3,1);
            epsi = var[(nrb+1)*ss_i-nrb+3];
                       
            p_ti=np.matmul(T_i,p_tendon); #position of tendon k at diski wrt i-1 frame
            p_i=np.matmul(T_i,np.array([0,0,0,1]).reshape(4,1)); #position of diski wrt i-1 frame
            norm_ct1=np.sqrt(np.sum(np.power((-1*p_ti[0:3,:]+p_tendon[0:3,:]),2),axis=0));
            Pi = np.zeros((4,nrb),dtype=int); # position of the rigid bodies
            for k in range(0,nrb)
                Pi[:,k] = Trb[:,3,k].reshape(4,1);
            
            # Tendon tension on the different disks
            # Tension of the first set of tendons apply on the proximal segment
            # only
            nt = (F.size)/(n.size); # Number of tendons per section
            Fdisk = [nummat.repmat(F,1,n[0]);[zeros(n(2),nt) repmat(F(nt+1:end),n(2),1)]];
            
            # Direction orthogonal to the disk
            zi = T_i(:,3,end);
            
            if ss_i<n_disk
                # Tendon force from disk ss_i to disk ss_i+1
                [T_i2,Trb] = trans_mat_prbm(var,nrb,gamma,l,ss_i+1,ss_i-1); #returns transformation matrix from i to i-1
                p_ti2=T_i2*p_tendon; #position of tendon k at diski wrt i-1 frame
                norm_ct2=sqrt(sum((-p_ti(1:3,:)+p_ti2(1:3,:)).^2));
                
                # Tendon force and moment: Eq (9)
                Fi = ((p_tendon-p_ti)./repmat(norm_ct1,4,1)).*repmat(Fdisk(ss_i,:),4,1)+((p_ti2-p_ti)./repmat(norm_ct2,4,1)).*repmat(Fdisk(ss_i+1,:),4,1);
                if ss_i==n(1)
                    # Tip of segment 1
                    # Consider the full force for tendon 1 to 3, remove
                    # component orthogonal to the disk for tendon 4 to 6
                    Fi = [Fi(:,1:3) Fi(:,4:6)-repmat((zi'*Fi(:,4:6)),[4,1]).*repmat(zi,[1,3])];
                else
                    # Remove component orthogonal to the disk for tendon 1
                    # to 6
                    Fi = Fi - repmat(zi'*Fi,[4,1]).*repmat(zi,[1,6]);
                end
            else
                Fi = ((p_tendon-p_ti)./repmat(norm_ct1,4,1)).*repmat(Fdisk(ss_i,:),4,1);
            end
            
            # Moment due to tendon force: Eq (12)
            Mi = cross_product(p_ti(1:3,:)-repmat(Pi(1:3,end),1,length(F)),Fi(1:3,:));
            
            # External forces and moments
            Rt = trans_mat_prbm(var,nrb,gamma,l,ss_i-1,0);
            Fex = Rt\Ftex; 

            R_ex = trans_mat_prbm(var,nrb,gamma,l,n_disk,ss_i-1);
            p_ex = R_ex(1:3,4);
            Mex = Rt(1:3,1:3)\Mtex(1:3)-cross_product(Pi(1:3,end)-p_ex,Fex(1:3));#+cross_product(p_i(1:3),Fex(1:3));      
            
            
            # Total forces and moments: Eq (17-18)
            if ss_i < n_disk
                # Tip of segment 1
                Ftot = T_i(1:3,1:3)*F_prev + sum(Fi(1:3,:),2);
                Mtot = T_i(1:3,1:3)*M_prev + cross_product(T_i2(1:3,4)-Pi(1:3,end),T_i(1:3,1:3)*F_prev) + sum(Mi,2);
            else 
                # Tip of segment 2
                Ftot =  sum(Fi(1:3,:),2);
                Mtot = sum(Mi,2);
            end
            
            # Bending stiffness at each joint
            K = [3.25*E*I/l(ss_i) 2.84*E*I/l(ss_i) 2.95*E*I/l(ss_i)];

            for k=1:nrb-1
                # Static equilibrium
                Rb = Trb(1:3,1:3,k+1);
                Mnetb = Rb'*(cross_product(Pi(1:3,end)-Pi(1:3,k),Ftot(1:3)+Fex(1:3))+Mtot+Mex);
                res((nrb+1)*ss_i-nrb+k-1)= K(k)*theta(k) - Mnetb(2);
            end
            
            # Geometrical constraint for determining phi
            Mnet = cross_product(p_i(1:3),Ftot(1:3)+Fex(1:3))+Mtot+Mex;  # Net moment at disk i in frame i
            Mphi = Mnet; 
            Mphi(3)=0;
            res((nrb+1)*ss_i-(nrb)+3) = ni'*(Mphi)-norm(Mphi);
            
            # Torsion
            Ri = T_i(1:3,1:3);
            Mepsi = Ri'*Mnet;
            res((nrb+1)*ss_i-(nrb)+4) = Mepsi(3)-2*G*I/l(ss_i)*epsi;
            
            F_prev = Ftot(1:3);
            M_prev = Mtot;

        end
    end
end

