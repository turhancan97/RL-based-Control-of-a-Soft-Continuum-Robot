
% This code implements different approaches to model the kinematics/statics
% of a two segment tendon driven continuum robot and is part of the following
% publication:

% How to model tendon-driven continuum robots and benchmark modelling performance
% Priyanka Rao, Quentin Peyron, Sven Lilge, Jessica Burgner-Kahrs
% frontiers in Robotics and AI 2021
% DOI: 10.3389/frobt.2020.630245

% Copyright (C) 2021 Continuum Robotics Laboratory, University of Toronto Mississauga

function [s,p] = construct_tdcr_prbm(nrb,gamma,l,n_disk,var)

%% visualisation of the results
[T_i,Trb]=trans_mat_prbm(var,nrb,gamma,l,n_disk,0);
p = [0;0;0];
s=0;
for k=1:n_disk
    for t=1:nrb
        p = [p Trb(1:3,4,nrb*(k-1)+t)];
        s = [s s(end)+gamma(t)*l(k)];
    end
end