# Libraries and important folders
import sys
sys.path.append('../')
sys.path.append('../Reinforcement Learning')
sys.path.append('../Pytorch')

import torch
import matplotlib.pyplot as plt
from ddpg_agent import Agent
from env import continuumEnv
import math
import time

env = continuumEnv()
env.seed(10)
agent = Agent(state_size=4, action_size=3, random_seed=10)

# %% Evaluation
#### Change the directory for your file structure
agent.actor_local.load_state_dict(torch.load('../Pytorch/model/checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('../Pytorch/model/checkpoint_critic.pth'))
 
state = env.reset() # generate random starting point for the robot and random target point.
env.start_kappa = [env.kappa1, env.kappa2, env.kappa3] # save starting kappas
initial_state = state[0:2]
x_pos = []
y_pos = []

for t in range(1000):
    start = time.time()
    action = agent.act(state, add_noise=False)
    state, reward, done, _ = env.step_2(action)
    x_pos.append(state[0])
    y_pos.append(state[1])
    # env.render() # uncomment for instant animation
    print("{}th action".format(t))
    print("Goal Position",state[2:4])
    print("Error: {0}, Current State: {1}".format(math.sqrt(-1*reward), state)) # for step_2
    print("Action: {0},  Kappas {1}".format(action, [env.kappa1,env.kappa2,env.kappa3]))
    print("Reward is ", reward)
    print("Episodic Reward is {}".format(reward))
    print("--------------------------------------------------------------------------------")
    stop = time.time()
    env.time += (stop - start)
    if done:
        break

# %% Visualization
env.visualization(x_pos,y_pos)
plt.title(f"Initial Position is x: {initial_state[0]} y: {initial_state[1]} & Target Position is x: {state[0]} y: {state[1]}")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.show()
env.close()