
% This code implements different approaches to model the kinematics/statics
% of a two segment tendon driven continuum robot and is part of the following
% publication:

% How to model tendon-driven continuum robots and benchmark modelling performance
% Priyanka Rao, Quentin Peyron, Sven Lilge, Jessica Burgner-Kahrs
% frontiers in Robotics and AI 2021
% DOI: 10.3389/frobt.2020.630245

% Copyright (C) 2021 Continuum Robotics Laboratory, University of Toronto Mississauga

function [T,Trb] = trans_mat_prbm(var,nrb,gamma,l,q,p)
%%retruns rotation matrix from frame q to p

T=eye(4);
Trb = zeros(4,4,nrb*(q-p)); % trnasformation matrix from i-1 disk to Pj

if q<p
    error('Some error in rotation matrix indices')
else
    for iter=p+1:1:q
        
%         thx = var((nrb+1)*(iter-1)+1);
%         thy = var((nrb+1)*(iter-1)+2);
%         th = sqrt(thx^2+thy^2);
%         theta = [th-sum(var((nrb+1)*(iter-1)+3:(nrb+1)*(iter-1)+4)) var((nrb+1)*(iter-1)+3) var((nrb+1)*(iter-1)+4)];
%         epsi = var((nrb+1)*(iter-1)+5);
%         phi = atan2(thy,thx);
        
        theta=var((nrb+1)*iter-nrb:(nrb+1)*iter-nrb+2);
        phi = var((nrb+1)*iter-nrb+3);
        epsi = var((nrb+1)*iter-nrb+4);
            
            
        R_phi = [cos(phi) -sin(phi) 0 0;...
            sin(phi) cos(phi) 0 0;...
            0 0 1 0;...
            0 0 0 1];
        
        R_phi_epsi = [cos(epsi-phi) -sin(epsi-phi) 0 0;...
            sin(epsi-phi) cos(epsi-phi) 0 0;...
            0 0 1 0;...
            0 0 0 1];
        
        Ti = R_phi*[1 0 0 0;0 1 0 0;0 0 1 gamma(1)*l(iter);0 0 0 1];
        Trb(:,:,nrb*(iter-(p+1))+1) = T*Ti;
        for k = 1:nrb-1
            Ti = Ti*[cos(theta(k)) 0 sin(theta(k)) gamma(k+1)*l(iter)*sin(theta(k));...
             0 1 0 0;...
             -sin(theta(k)) 0 cos(theta(k)) gamma(k+1)*l(iter)*cos(theta(k));...
             0 0 0 1];
            Trb(:,:,nrb*(iter-(p+1))+k+1)=T*Ti;
        end
        T = T*Ti*R_phi_epsi;
            
    end
end
end
