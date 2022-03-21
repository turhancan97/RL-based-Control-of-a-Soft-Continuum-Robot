
% This code implements different approaches to model the kinematics/statics
% of a two segment tendon driven continuum robot and is part of the following
% publication:

% How to model tendon-driven continuum robots and benchmark modelling performance
% Priyanka Rao, Quentin Peyron, Sven Lilge, Jessica Burgner-Kahrs
% frontiers in Robotics and AI 2021
% DOI: 10.3389/frobt.2020.630245

% Copyright (C) 2021 Continuum Robotics Laboratory, University of Toronto Mississauga

function [R] = rotation_mat_prbm(var,l,q,p)
%%retruns rotation matrix from frame q to p

R=eye(3);
if q<p
    error('Some error in rotation matrix indices')
else
    for iter=p+1:1:q
        beta=var(3*iter-2);
        gamma=var(3*iter-1);
        epsi=var(3*iter);
        k=sqrt(beta^2+gamma^2);
        phi=atan2(gamma,beta);
        theta=k*l(iter);
        
        Rz=[cos(phi) -sin(phi) 0;
            sin(phi) cos(phi) 0;
            0 0 1];
        Ry=[cos(theta) 0 sin(theta);
            0 1 0;
            -sin(theta) 0 cos(theta)];
        Rz2=[cos(epsi-phi) -sin(epsi-phi) 0;
            sin(epsi-phi) cos(epsi-phi) 0;
            0 0 1];
        R=R*Rz*Ry*Rz2;
    end
end
end