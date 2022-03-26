from cross_product import cross_product

# function [var,exitflag,res] = prbm_solver(n,nrb,gamma,l,F,p_tendon,m_disk,m_bb,E,I,G,Ftex,Mtex,var0)

def prbm_solver(n,nrb,gamma,l,F,p_tendon,m_disk,m_bb,E,I,G,Ftex,Mtex,var0):
    ##returns the solved values for [beta, gamma, epsi] for each subsegment i
    n_disk = np.sum(n);
    #options1 = optimset('Display','iter','TolFun',1e-6,'MaxFunEvals',1500,'TolX',1e-6,'Algorithm','trust-region-dogleg'); 

    #tic
    #var,infodict = fsolve(optim_f,var0);
    #toc

## solver

    #function [res] = optim_f(var)
    def optim_f(var):
        
        res=np.zeros((n_disk*(nrb+1),1)); #nrb-1 revolute joints, 1 bending plane angle and 1 torsion angle
        F_prev = np.zeros((3,1));
        M_prev = np.zeros((3,1));
        for ss_i in range(n_disk-1,-1,-1) #iterating over each subsegment
            
            # Kinematics
            [T_i,Trb] = trans_mat_prbm(var,nrb,gamma,l,ss_i,ss_i-1); #returns transformation matrix from i to i-1
            
            theta=var[0][((nrb+1)*(ss_i+1)-nrb-1):(nrb+1)*(ss_i+1)-nrb+2];
            phi = var((nrb+1)*ss_i-nrb+3);
            ni = [cos(phi+pi/2);sin(phi+pi/2);0];
            epsi = var((nrb+1)*ss_i-nrb+4);
            
            p_ti=T_i*p_tendon; #position of tendon k at diski wrt i-1 frame
            p_i=T_i*[0;0;0;1]; #position of diski wrt i-1 frame
            norm_ct1=sqrt(sum((-p_ti(1:3,:)+p_tendon(1:3,:)).^2));
            Pi = zeros(4,nrb); # position of the rigid bodies
            for k=1:nrb
                Pi(:,k) = Trb(:,4,k);
            end
            
            # Tendon tension on the different disks
            # Tension of the first set of tendons apply on the proximal segment
            # only
            nt = length(F)/length(n); # Number of tendons per section
            Fdisk = [repmat(F,n(1),1);[zeros(n(2),nt) repmat(F(nt+1:end),n(2),1)]];
            
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