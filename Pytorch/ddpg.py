import sys
sys.path.append('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Reinforcement Learning')

# import gym
# import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent
from env import continuumEnv
import time
import math

env = continuumEnv()
env.seed(10)
agent = Agent(state_size=4, action_size=3, random_seed=10)

# %%
def ddpg(n_episodes=500, max_t=750, print_every=1):
    global scores
    global avg_reward_list
    scores_deque = deque(maxlen=print_every)
    scores = []
    avg_reward_list = []
    counter = 0
    for i_episode in range(1, n_episodes+1):
        state = env.reset_unknown()
        agent.reset()
        score = 0
        print("Initial Position is",state[0:2])
        print("===============================================================")
        print("Target Position is",state[2:4])
        print("===============================================================")
        print("Initial Kappas are ",[env.kappa1,env.kappa2,env.kappa3])
        print("===============================================================")
        print("Goal Kappas are ",[env.target_k1,env.target_k2,env.target_k3])
        print("===============================================================")
        time.sleep(2)
        
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step_2(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            print("Episode Number {0} and {1}th action".format(i_episode,t))
            print("Goal Position",state[2:4])
            # print("Previous Error: {0}, Error: {1}, Current State: {2}".format(env.previous_error, env.error, prev_state)) # for step_1
            print("Error: {0}, Current State: {1}".format(math.sqrt(-1*reward), state)) # for step_2
            print("Action: {0},  Kappas {1}".format(action, [env.kappa1,env.kappa2,env.kappa3]))
            print("Reward is ", reward)
            print("{0} times robot reached to the target".format(counter))
            print("Avg Reward is {0}, Episodic Reward is {1}".format(np.mean(scores),score))
            print("--------------------------------------------------------------------------------")
            score += reward
            if done:
                counter += 1
                break 
        scores_deque.append(score)
        scores.append(score)
        avg_reward_list.append(np.mean(scores))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        time.sleep(2)
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            time.sleep(2)
            
    return scores

scores = ddpg()

# %% Result
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Reward')
plt.xlabel('Episode #')
plt.subplot(1, 2, 2)
plt.plot(np.arange(1, len(avg_reward_list)+1), avg_reward_list)
plt.ylabel('Average Reward')
plt.xlabel('Episode #')
plt.show()
