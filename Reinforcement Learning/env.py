import sys
sys.path.append('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Functions')
sys.path.append('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Functions')

import gym
import numpy as np
import math
# import random
# from numpy import sin, cos, pi
from gym import spaces
import matplotlib.pyplot as plt
# import time
from forward_velocity_kinematics import three_section_planar_robot, jacobian_matrix
from forward_velocity_kinematics import trans_mat_cc, coupletransformations

class continuumEnv(gym.Env):
    """
    Class docstrings go here. # TODO: add nice description for the environment.
    * A brief summary of its purpose and behavior
    * Any public methods, along with a brief description - e.g. step
    * Any class properties (attributes) e.g. kappa_dot_max, dt
    * Anything related to the interface for subclassers, if the class is intended to be subclassed
    
    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says
    name : str
        the name of the animal
    sound : str
        the sound that the animal makes
    num_legs : int
        the number of legs the animal has (default 4)

    Methods
    -------
    step()
        Prints the animals name and what sound it makes
        
    """
    
    def __init__(self):

        # self.viewer = None
        self.delta_kappa = 0.1
        self.kappa_dot_max = 1.000 # 1.00; # derivative of curvature
        self.kappa_max = 16.00
        self.kappa_min = -4.00
        self.q_goal = 0 # case 3 Goal Position
        #self.q_goal = np.array([-0.186, 0.1995]) #case 1 and case 2
        # self.kappa1 = 0.50 # initial kappa 1
        # self.kappa2 = 0.50 # initial kappa 2
        # self.kappa3 = 0.50 # initial kappa 3
        l1 = 0.1000; # metre
        l2 = 0.1000; # metre
        l3 = 0.1000; # metre
        self.stop = 0 # variable to make robot not move after exeeding max, min general kappa value
        # self.stop1 = 0 # variable to make robot not move after exeeding max, min kappa1 value
        # self.stop2 = 0 # variable to make robot not move after exeeding max, min kappa2 value
        # self.stop3 = 0 # variable to make robot not move after exeeding max, min kappa3 value
        self.l = [l1, l2, l3] # Each section's length
        self.dt =  1e-1 # sample sizes
        self.J = np.zeros((2,3))
        self.error = 0
        self.previous_error = 0
        self.start_kappa = [0,0,0]
        high = np.array([0.2, 0.3, 0.16, 0.3], dtype=np.float32) # [0.16, 0.3, 0.16, 0.3]
        low =   np.array([-0.3, -0.15, -0.27, -0.11], dtype=np.float32) # [-0.27, -0.11, -0.27, -0.11]
        self.action_space = spaces.Box(low=-1*self.kappa_dot_max, high=self.kappa_dot_max,shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
    def step_1(self,u): # reward is -1 or 0 or 1
        
        x,y = self.state
        global new_x
        global new_y
        # delta_kappa = self.delta_kappa
        # l = self.l
        # kappa1 = self.kappa1
        # kappa2 = self.kappa2
        # kappa3 = self.kappa3
        dt =  self.dt # sample sizes
        
        u = np.clip(u, -self.kappa_dot_max, self.kappa_dot_max)[0]
        
        self.error = math.sqrt((((self.q_goal[0]-self.state[0])**2)+((self.q_goal[1]-self.state[1])**2)))
        
        
        if self.error < self.previous_error:
            self.costs = 1
        elif self.error == self.previous_error:
            self.costs = -0.5
        else:
            self.costs = -1
        
        self.previous_error = self.error
        
        
        if self.error <= 0.015: # or 0.01
            done = True
        else :
            done = False
         
        
        if self.stop == 0:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ u
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 1:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append([0],u[1:3])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 2:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(np.append(u[0],[0]),u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 3:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(u[0:2],[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 4:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append([0,0],u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 5:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @  np.append(np.append([0],u[1]),[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 6:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(u[0],[0,0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 7:
            print("Robot is not moving")
            # time.sleep(1)
        
        self.kappa1 += u[0] * dt # TODO -> prevent to be different than -4,16
        self.kappa2 += u[1] * dt
        self.kappa3 += u[2] * dt
        
        # TODO -> Solve the situation when kappas are zero in Homogenous matrix
        self.kappa1 = np.clip(self.kappa1, self.kappa_min, self.kappa_max)
        self.kappa2 = np.clip(self.kappa2, self.kappa_min, self.kappa_max)
        self.kappa3 = np.clip(self.kappa3, self.kappa_min, self.kappa_max)
        
        self.stop = 0
        # self.stop1 = 0
        # self.stop2 = 0
        # self.stop3 = 0
        k1 = self.kappa1 == self.kappa_min or self.kappa1 == self.kappa_max
        k2 = self.kappa2 == self.kappa_min or self.kappa2 == self.kappa_max
        k3 = self.kappa3 == self.kappa_min or self.kappa3 == self.kappa_max
        
        if k1:
            self.stop = 1
            
        elif k2:
            self.stop = 2
            
        elif k3:
            self.stop = 3
        
        if k1 and k2:
            self.stop = 4
        
        elif k1 and k3:
            self.stop = 5
        
        elif k2 and k3:
            self.stop = 6
            
        if k1 and k2 and k3:
            self.stop = 7

        new_x = np.clip(new_x, self.observation_space.low[0], self.observation_space.high[0])
        new_y = np.clip(new_y, self.observation_space.low[1], self.observation_space.high[1])
        
        self.state = np.array([new_x,new_y])
        
        return self._get_obs(), self.costs, done, {}
    
    def step_2(self,u): # reward is -(e^2)
        
        x,y,goal_x,goal_y = self.state
        global new_x
        global new_y
        global new_goal_x
        global new_goal_y
        # delta_kappa = self.delta_kappa
        # l = self.l
        # kappa1 = self.kappa1
        # kappa2 = self.kappa2
        # kappa3 = self.kappa3
        dt =  self.dt # sample sizes
        
        u = np.clip(u, -self.kappa_dot_max, self.kappa_dot_max)
        
        self.error = ((goal_x-x)**2)+((goal_y-y)**2)
        self.costs = self.error
        
        if self.error < self.previous_error:
            #self.costs -= 1
            print("=========================POSITIVE MOVE=========================")
            
        
        self.previous_error = self.error
        #     self.costs = 1 - self.error
        # elif self.error == self.previous_error:
        #     self.costs = -0.5 - self.error
        # else:
        #     self.costs = -1 - self.error
            
        # if self.error < self.previous_error and self.error <= 0.04: # or 0.01
        #     self.costs = 10 - self.error
        # elif self.error < self.previous_error and self.error <= 0.05: # or 0.01
        #     self.costs = 9 - self.error
        # elif self.error < self.previous_error and self.error <= 0.06: # or 0.01
        #     self.costs = 8 - self.error
        # elif self.error < self.previous_error and self.error <= 0.07: # or 0.01
        #     self.costs = 7 - self.error
        # elif self.error < self.previous_error and self.error <= 0.08: # or 0.01
        #     self.costs = 6 - self.error
        
        # self.previous_error = self.error
        
        if math.sqrt(self.costs) <= 0.01: # or 0.01
            done = True
        else :
            done = False
         
        
        if self.stop == 0:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ u
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 1:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append([0],u[1:3])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 2:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(np.append(u[0],[0]),u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 3:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(u[0:2],[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 4:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append([0,0],u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 5:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @  np.append(np.append([0],u[1]),[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 6:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(u[0],[0,0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 7:
            print("Robot is not moving")
            # time.sleep(1)
        
        self.kappa1 += u[0] * dt # TODO -> prevent to be different than -4,16
        self.kappa2 += u[1] * dt
        self.kappa3 += u[2] * dt
        
        # TODO -> Solve the situation when kappas are zero in Homogenous matrix
        self.kappa1 = np.clip(self.kappa1, self.kappa_min, self.kappa_max)
        self.kappa2 = np.clip(self.kappa2, self.kappa_min, self.kappa_max)
        self.kappa3 = np.clip(self.kappa3, self.kappa_min, self.kappa_max)
        
        self.stop = 0
        # self.stop1 = 0
        # self.stop2 = 0
        # self.stop3 = 0
        k1 = self.kappa1 <= self.kappa_min or self.kappa1 >= self.kappa_max
        k2 = self.kappa2 <= self.kappa_min or self.kappa2 >= self.kappa_max
        k3 = self.kappa3 <= self.kappa_min or self.kappa3 >= self.kappa_max
        
        if k1:
            self.stop = 1
            
        elif k2:
            self.stop = 2
            
        elif k3:
            self.stop = 3
        
        if k1 and k2:
            self.stop = 4
        
        elif k1 and k3:
            self.stop = 5
        
        elif k2 and k3:
            self.stop = 6
            
        if k1 and k2 and k3:
            self.stop = 7

        new_x = np.clip(new_x, self.observation_space.low[0], self.observation_space.high[0])
        new_y = np.clip(new_y, self.observation_space.low[1], self.observation_space.high[1])
        new_goal_x = np.clip(goal_x, self.observation_space.low[2], self.observation_space.high[2])
        new_goal_y = np.clip(goal_y, self.observation_space.low[3], self.observation_space.high[3])
        
        self.state = np.array([new_x,new_y,new_goal_x,new_goal_y])
        
        return self._get_obs(), -1*self.costs, done, {}
   
    def reset(self):
       
       # Random state for beginning
       self.kappa1 = np.random.uniform(low=-4, high=16)
       self.kappa2 = np.random.uniform(low=-4, high=16)
       self.kappa3 = np.random.uniform(low=-4, high=16)
       
       
       T3_cc = three_section_planar_robot(self.kappa1, self.kappa2, self.kappa3, self.l)
       x,y = np.array([T3_cc[0,3],T3_cc[1,3]])
       
       # Random target point
       self.target_k1 = np.random.uniform(low=-4, high=16)
       self.target_k2 = np.random.uniform(low=-4, high=16)
       self.target_k3 = np.random.uniform(low=-4, high=16)
       
       T3_target = three_section_planar_robot(self.target_k1,self.target_k2,self.target_k3, self.l)
       goal_x,goal_y = np.array([T3_target[0,3],T3_target[1,3]])
       
       self.state = x,y,goal_x,goal_y
       
       self.last_u = None
       return self._get_obs()
    
    def _get_obs(self):
        x,y,goal_x,goal_y = self.state
        return np.array([x,y,goal_x,goal_y],dtype=np.float32)
    
    def render(self,x_pos,y_pos):
        # If you do not want to see the whole robot in the plot uncomment this
        T1_cc = trans_mat_cc(self.start_kappa[0],self.l[0])
        T1_tip = np.reshape(T1_cc[len(T1_cc)-1,:],(4,4),order='F');
        # section 2
        T2 = trans_mat_cc(self.start_kappa[1],self.l[1]);
        T2_cc = coupletransformations(T2,T1_tip);
        T2_tip = np.reshape(T2_cc[len(T2_cc)-1,:],(4,4),order='F');
        # section 3
        T3 = trans_mat_cc(self.start_kappa[2],self.l[2]);
        T3_cc = coupletransformations(T3,T2_tip);
        # Plot the trunk with three sections and point the section seperation

        plt.plot([-0.025, 0.025],[0,0],'black',linewidth=5)
        plt.plot(T1_cc[:,12],T1_cc[:,13],'b',linewidth=3)
        #plt.scatter(T1_cc[-1,12],T1_cc[-1,13],linewidths=5,color = 'black')
        plt.plot(T2_cc[:,12],T2_cc[:,13],'r',linewidth=3)
        #plt.scatter(T2_cc[-1,12],T2_cc[-1,13],linewidths=5,color = 'black')
        plt.plot(T3_cc[:,12],T3_cc[:,13],'g',linewidth=3)
        plt.scatter(T3_cc[-1,12],T3_cc[-1,13],linewidths=5,color = 'black')

        T1_cc = trans_mat_cc(self.kappa1,self.l[0])
        T1_tip = np.reshape(T1_cc[len(T1_cc)-1,:],(4,4),order='F');
        # section 2
        T2 = trans_mat_cc(self.kappa2,self.l[1]);
        T2_cc = coupletransformations(T2,T1_tip);
        T2_tip = np.reshape(T2_cc[len(T2_cc)-1,:],(4,4),order='F');
        # section 3
        T3 = trans_mat_cc(self.kappa3,self.l[2]);
        T3_cc = coupletransformations(T3,T2_tip);

        # Plot the trunk with three sections and point the section seperation
        plt.plot(T1_cc[:,12],T1_cc[:,13],'b',linewidth=3)
        #plt.scatter(T1_cc[-1,12],T1_cc[-1,13],linewidths=5,color = 'black')
        plt.plot(T2_cc[:,12],T2_cc[:,13],'r',linewidth=3)
        #plt.scatter(T2_cc[-1,12],T2_cc[-1,13],linewidths=5,color = 'black')
        plt.plot(T3_cc[:,12],T3_cc[:,13],'g',linewidth=3)
        plt.scatter(T3_cc[-1,12],T3_cc[-1,13],linewidths=5,color = 'black')        
        
        plt.scatter(self.state[2],self.state[3],100, marker= "x",linewidths=2, color = 'red')
        plt.scatter(x_pos,y_pos,25,linewidths=0.01,color = 'blue',alpha=0.1)
        plt.xlim([-0.4, 0.4])
        plt.ylim([-0.4, 0.4])