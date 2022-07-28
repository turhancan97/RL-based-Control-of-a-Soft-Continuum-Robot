# Libraries and important folders
import sys
sys.path.append('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Reinforcement Learning')
sys.path.append('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Pytorch')

import torch
import matplotlib.pyplot as plt
from ddpg_agent import Agent
from env import continuumEnv
import math

env = continuumEnv()
env.seed(10)
agent = Agent(state_size=4, action_size=3, random_seed=10)

# %% Evaluation
#### Change the directory for your file structure
agent.actor_local.load_state_dict(torch.load('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Pytorch/Weights/checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Pytorch/Weights/checkpoint_critic.pth'))
 
state = env.reset() # generate random starting point for the robot and random target point.
env.start_kappa = [env.kappa1, env.kappa2, env.kappa3] # save starting kappas

x_pos = []
y_pos = []

for t in range(750):
    action = agent.act(state, add_noise=False)
    state, reward, done, _ = env.step_2(action)
    x_pos.append(state[0])
    y_pos.append(state[1])
    print("{}th action".format(t))
    print("Goal Position",state[2:4])
    print("Error: {0}, Current State: {1}".format(math.sqrt(-1*reward), state)) # for step_2
    print("Action: {0},  Kappas {1}".format(action, [env.kappa1,env.kappa2,env.kappa3]))
    print("Reward is ", reward)
    print("Episodic Reward is {}".format(reward))
    print("--------------------------------------------------------------------------------")
    if done:
        break

# %% Visualization
env.render(x_pos,y_pos)
plt.title("Trajectory of the Continuum Robot")
plt.xlabel("X - Position")
plt.ylabel("Y - Position")
plt.show()
env.close()