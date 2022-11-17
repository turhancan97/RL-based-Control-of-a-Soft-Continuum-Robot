'''
    Author: Turhan Can KARGIN
    Python Version: 3.9.7
    Environment for the Continuum Robot
'''
# %% import necessary libraries
import sys # to include the path of the package
sys.path.append('../') # the kinematics functions are here 

import gym                      # openai gym library
import numpy as np              # numpy for matrix operations
import math                     # math for basic calculations
from gym import spaces          # "spaces" for the observation and action space
import matplotlib.pyplot as plt # quick "plot" library

# My own libraries
from kinematics.forward_velocity_kinematics import three_section_planar_robot, jacobian_matrix # the velocity kinematics
from kinematics.forward_velocity_kinematics import trans_mat_cc, coupletransformations # forward kinematics

class continuumEnv(gym.Env):
    """
    ### Description

    Robots with a continuous “backbone” on the other hand, have a wide range of maneuverability and can have a huge number 
    of degrees of freedom. Unlike traditional robots, where motion happens in discrete points, 
    such as joints, continuum style robots generate motion by bending the robot over a specific segment.

    Our system's aim is to take the three segment continuum robot from a random starting point to a random target by using 
    the forward kinematics in (19) and velocity kinematics formulas in (24) in the article below.

    *-> Hannan, M. W. & Walker, I. D. Kinematics and the implementation of an elephant’s trunk manipulator and other 
    continuum style robots. J. Robot. Syst. 20, 45–63 (2003).
   
    -  `x-y`: cartesian coordinates of the robot's tip point in meters.
    - `kappa` : curvatures in 1/m.
    - `kappa_dot` : derivative of curvatures in 1/m/s.

    ### Action Space
    The action is a `ndarray` with shape `(3,)` representing the derivative of each segment's curvature.
    
    | Num | Action  | Min  | Max |
    |-----|---------|------|-----|
    | 0   | K_dot_1 | -1.0 | 1.0 |
    | 1   | K_dot_2 | -1.0 | 1.0 |
    | 2   | K_dot_3 | -1.0 | 1.0 |

    ### Observation Space
    The observation is a `ndarray` with shape `(4,)` representing the x-y coordinates of the robot's starting and end points.
    
    | Num | Observation | Min   | Max   |
    |-----|-------------|-------|-------|
    | 0   | x           | -0.3  | 0.2   |
    | 1   | y           | -0.15 | 0.3   |
    | 2   | x_goal      | -0.27 | 0.16  |
    | 3   | y_goal      | -0.11 | 0.3   |
        

    ### Rewards
    The reward function is defined as:

    -> r = -(((x-x_goal)^2)-((y-y_goal)^2)) = -(e^2) <-
    which means that the reward is negative of the squared distance between the robot's tip point and the target point.

    * Environment is where the agent resides and is connected to, the environment is such that the agent can learn and interact.
    The environment in which the agent is located is partially observable or fully observable. In reinforcement learning, 
    the environment and observations of the agent can be random. Because there is no open access environment for the continuum robot,
    this class is created to simulate the environment. The environment is created by using the forward kinematics and velocity kinematics.

    * There are several methods in this class. The first method is the reset method. This method is used to reset the environment.
    * There are several attributes in this class. For example some of the attribute is kappa_dot_max, kappa_max, and kappa_min
    which are the maximum and minimum values of the curvature and the maximum value of the curvature derivative. Thanks to these attributes,
    we can limit the actions and movement of the robot.
    * Furter more details can be found in comments in the code.
        
    """
    
    def __init__(self):

        self.delta_kappa = 0.1      # necessary for the numerical differentiation
        self.kappa_dot_max = 1.000  # max derivative of curvature
        self.kappa_max = 16.00      # max curvature for the robot
        self.kappa_min = -4.00      # min curvature for the robot
        # self.q_goal = 0 # case 3 Goal Position
        # self.q_goal = np.array([-0.186, 0.1995]) #case 1 and case 2
        # self.kappa1 = 0.50 # initial kappa 1
        # self.kappa2 = 0.50 # initial kappa 2
        # self.kappa3 = 0.50 # initial kappa 3
        l1 = 0.1000;                # first segment of the robot in meters
        l2 = 0.1000;                # second segment of the robot in meters
        l3 = 0.1000;                # third segment of the robot in meters
        self.stop = 0               # variable to make robot not move after exeeding max, min general kappa value
        # self.stop1 = 0 # variable to make robot not move after exeeding max, min kappa1 value
        # self.stop2 = 0 # variable to make robot not move after exeeding max, min kappa2 value
        # self.stop3 = 0 # variable to make robot not move after exeeding max, min kappa3 value
        self.l = [l1, l2, l3]       # stores the length of each segment of the robot
        self.dt =  1e-1             # sample sizes
        self.J = np.zeros((2,3))    # initializes the Jacobian matrix  
        self.error = 0              # initializes the error
        self.previous_error = 0     # initializes the previous error
        self.start_kappa = [0,0,0]  # initializes the start kappas for the three segments
        self.time = 0               # to count the time of the simulation
        
        # Define the observation and action space from OpenAI Gym
        high = np.array([0.2, 0.3, 0.16, 0.3], dtype=np.float32) # [0.16, 0.3, 0.16, 0.3]
        low  = np.array([-0.3, -0.15, -0.27, -0.11], dtype=np.float32) # [-0.27, -0.11, -0.27, -0.11]
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
        
        x,y,goal_x,goal_y = self.state # Get the current state as x,y,goal_x,goal_y
        
        # global variables to be used in the reward function
        global new_x 
        global new_y
        global new_goal_x
        global new_goal_y

        # delta_kappa = self.delta_kappa
        # l = self.l
        # kappa1 = self.kappa1
        # kappa2 = self.kappa2
        # kappa3 = self.kappa3
        dt =  self.dt # Time step
        
        u = np.clip(u, -self.kappa_dot_max, self.kappa_dot_max) # Clip the input to the range of the -1,1
        
        self.error = ((goal_x-x)**2)+((goal_y-y)**2) # Calculate the error squared
        self.costs = self.error # Set the cost (reward) to the error squared
        
        # Just to show if the robot is moving along the goal or not
        if self.error < self.previous_error:
            #self.costs -= 1
            # UNCOMMENT HERE !!!!!!!
            print("=========================POSITIVE MOVE=========================")
            
        
        self.previous_error = self.error # Update the previous error
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
        
        # if the error is less than 0.01, the robot is close to the goal and returns done
        if math.sqrt(self.costs) <= 0.01:
            done = True
        else :
            done = False
         
        
        # This if and else statement is to avoid the robot to move if the kappas are at the limits
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
            # # UNCOMMENT HERE!!!!!!!
            print("Robot is not moving")
            # time.sleep(1)
        
        # Update the curvatures
        self.kappa1 += u[0] * dt 
        self.kappa2 += u[1] * dt
        self.kappa3 += u[2] * dt
        
        # TODO -> Solve the situation when kappas are zero in Homogenous matrix
        # Maybe when it is Zero try except and Raise an error
        self.kappa1 = np.clip(self.kappa1, self.kappa_min, self.kappa_max)
        self.kappa2 = np.clip(self.kappa2, self.kappa_min, self.kappa_max)
        self.kappa3 = np.clip(self.kappa3, self.kappa_min, self.kappa_max)
        
        # To check which curvature value are at the limits
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

        # Clip the states to avoid the robot to go out of the workspace
        new_x = np.clip(new_x, self.observation_space.low[0], self.observation_space.high[0])
        new_y = np.clip(new_y, self.observation_space.low[1], self.observation_space.high[1])
        new_goal_x = np.clip(goal_x, self.observation_space.low[2], self.observation_space.high[2])
        new_goal_y = np.clip(goal_y, self.observation_space.low[3], self.observation_space.high[3])
        
        # States of the robot in numpy array
        self.state = np.array([new_x,new_y,new_goal_x,new_goal_y])
        
        return self._get_obs(), -1*self.costs, done, {} # Return the observation, the reward (-costs) and the done flag
   
    def reset(self):
       
       # Random state of the robot 
       # (Random curvatures are given so that forward kinematics equation will generate random starting position)
       self.kappa1 = np.random.uniform(low=-4, high=16)
       self.kappa2 = np.random.uniform(low=-4, high=16)
       self.kappa3 = np.random.uniform(low=-4, high=16)
    
       T3_cc = three_section_planar_robot(self.kappa1, self.kappa2, self.kappa3, self.l) # Generate the position of the tip of the robot
       x,y = np.array([T3_cc[0,3],T3_cc[1,3]]) # Extract the x and y coordinates of the tip
       
       # Random target point
       # (Random curvatures are given so that forward kinematics equation will generate random target position)
       self.target_k1 = np.random.uniform(low=-4, high=16)
       self.target_k2 = np.random.uniform(low=-4, high=16)
       self.target_k3 = np.random.uniform(low=-4, high=16)
       
       T3_target = three_section_planar_robot(self.target_k1,self.target_k2,self.target_k3, self.l) # Generate the target point for the robot
       goal_x,goal_y = np.array([T3_target[0,3],T3_target[1,3]]) # Extract the x and y coordinates of the target
       
       self.state = x,y,goal_x,goal_y # Update the state of the robot
       
       self.last_u = None
       return self._get_obs()
    
    def _get_obs(self):
        x,y,goal_x,goal_y = self.state
        return np.array([x,y,goal_x,goal_y],dtype=np.float32)
    
    def render(self):
        # This function is used to plot the robot in the environment (both in start and end state)

        # current state
        # section 1 calculation
        T1_cc = trans_mat_cc(self.kappa1,self.l[0])
        T1_tip = np.reshape(T1_cc[len(T1_cc)-1,:],(4,4),order='F');
        # section 2 calculation
        T2 = trans_mat_cc(self.kappa2,self.l[1]);
        T2_cc = coupletransformations(T2,T1_tip);
        T2_tip = np.reshape(T2_cc[len(T2_cc)-1,:],(4,4),order='F');
        # section 3 calculation
        T3 = trans_mat_cc(self.kappa3,self.l[2]);
        T3_cc = coupletransformations(T3,T2_tip);

        # Plot the trunk with three sections and point the section seperation
        plt.plot([-0.025, 0.025],[0,0],'black',linewidth=5)
        plt.plot(T1_cc[:,12],T1_cc[:,13],'b',linewidth=3)
        #plt.scatter(T1_cc[-1,12],T1_cc[-1,13],linewidths=5,color = 'black')
        plt.plot(T2_cc[:,12],T2_cc[:,13],'r',linewidth=3)
        #plt.scatter(T2_cc[-1,12],T2_cc[-1,13],linewidths=5,color = 'black')
        plt.plot(T3_cc[:,12],T3_cc[:,13],'g',linewidth=3)
        plt.scatter(T3_cc[-1,12],T3_cc[-1,13],linewidths=5,color = 'black')        
        
        # Plot the target point and trajectory of the robot
        plt.scatter(self.state[2],self.state[3],100, marker= "x",linewidths=2, color = 'red')
        plt.title(f"The time elapsed in the simulation is {round(self.time,2)} seconds.")
        plt.xlabel("X - Position [m]")
        plt.ylabel("Y - Position [m]")
        plt.xlim([-0.4, 0.4])
        plt.ylim([-0.4, 0.4])
        plt.pause(0.000001)
        plt.clf()
        
        
    def visualization(self,x_pos,y_pos):
        # This function is used to plot the robot in the environment (both in start and end state)

        # Start state
        # section 1 calculation
        T1_cc = trans_mat_cc(self.start_kappa[0],self.l[0])
        T1_tip = np.reshape(T1_cc[len(T1_cc)-1,:],(4,4),order='F');
        # section 2 calculation
        T2 = trans_mat_cc(self.start_kappa[1],self.l[1]);
        T2_cc = coupletransformations(T2,T1_tip);
        T2_tip = np.reshape(T2_cc[len(T2_cc)-1,:],(4,4),order='F');
        # section 3 calculation
        T3 = trans_mat_cc(self.start_kappa[2],self.l[2]);
        T3_cc = coupletransformations(T3,T2_tip);

        # Plot the trunk with three sections and point the section seperation
        plt.plot([-0.025, 0.025],[0,0],'black',linewidth=5)
        plt.plot(T1_cc[:,12],T1_cc[:,13],'b',linewidth=3)
        #plt.scatter(T1_cc[-1,12],T1_cc[-1,13],linewidths=5,color = 'black')
        plt.plot(T2_cc[:,12],T2_cc[:,13],'r',linewidth=3)
        #plt.scatter(T2_cc[-1,12],T2_cc[-1,13],linewidths=5,color = 'black')
        plt.plot(T3_cc[:,12],T3_cc[:,13],'g',linewidth=3)
        plt.scatter(T3_cc[-1,12],T3_cc[-1,13],linewidths=5,color = 'black',label='Initial Point')

        # End state
        # section 1 calculation
        T1_cc = trans_mat_cc(self.kappa1,self.l[0])
        T1_tip = np.reshape(T1_cc[len(T1_cc)-1,:],(4,4),order='F');
        # section 2 calculation
        T2 = trans_mat_cc(self.kappa2,self.l[1]);
        T2_cc = coupletransformations(T2,T1_tip);
        T2_tip = np.reshape(T2_cc[len(T2_cc)-1,:],(4,4),order='F');
        # section 3 calculation
        T3 = trans_mat_cc(self.kappa3,self.l[2]);
        T3_cc = coupletransformations(T3,T2_tip);

        # Plot the trunk with three sections and point the section seperation
        plt.plot(T1_cc[:,12],T1_cc[:,13],'b',linewidth=3)
        #plt.scatter(T1_cc[-1,12],T1_cc[-1,13],linewidths=5,color = 'black')
        plt.plot(T2_cc[:,12],T2_cc[:,13],'r',linewidth=3)
        #plt.scatter(T2_cc[-1,12],T2_cc[-1,13],linewidths=5,color = 'black')
        plt.plot(T3_cc[:,12],T3_cc[:,13],'g',linewidth=3)
        plt.scatter(T3_cc[-1,12],T3_cc[-1,13],linewidths=5,color = 'black')        
        
        # Plot the target point and trajectory of the robot
        plt.scatter(self.state[2],self.state[3],100, marker= "x",linewidths=4, color = 'red',label='Target Point')
        plt.scatter(x_pos,y_pos,25,linewidths=0.03,color = 'blue',alpha=0.2)
        plt.xlim([-0.4, 0.4])
        plt.ylim([-0.4, 0.4])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(which='major',linewidth=0.7)
        plt.grid(which='minor',linewidth=0.5)
        # Show the minor ticks and grid.
        plt.minorticks_on()
        