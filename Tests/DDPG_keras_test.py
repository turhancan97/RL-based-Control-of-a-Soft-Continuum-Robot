# %%
import sys

sys.path.append('../')
sys.path.append('../Reinforcement Learning')
sys.path.append('../Keras')

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random
from env import continuumEnv
from DDPG import OUActionNoise, policy

env = continuumEnv()

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(3), std_deviation=float(std_dev) * np.ones(3))


# %% Evaluate
# Some of the initial and target states. Uncomment below if you want to see behavior of the agent for theses states.
# Otherwise, the agent will be evaluated for random states.
# t = random.randint(0,10)
# env.kappa1 = [-2,2,14.108017793675163,1.7150558451751081,-4,16,0.01,14,0.01,2.45,-3.374][t]
# env.kappa2 = [-3,-1,9.135168696509695,11.867430716497964,-4,16,0.01,14,12,13.58,15.9254][t]
# env.kappa3 = [10,-0.8133168825984081,-1.6793285894745877,5.6148372497202175,-4,16,0.01,14,10,12.64,11.91][t]
# env.target_k1 = [12,6,3.4625760406944948,0.29615171650441585,16,-4,-3,-2,-1,6.842,10.405][t]
# env.target_k2 = [12,-3,-4.0,4.712192928536007,16,-4,-3,-2,-2,-4.0,16.0][t]
# env.target_k3 = [16,-1,-4.,2.8197409208656574,16,-4,-3,-2,-3,-4.0,11.824][t]

state = env.reset() # generate random starting point for the robot and random target point.
env.start_kappa = [env.kappa1, env.kappa2, env.kappa3] # save starting kappas
initial_state = state[0:2]
error_store = [] # store the error value here
x_pos = [] # store the x position value here
y_pos = [] # store the y position here
# kappa1_store = [] # store kappa 1 values
# kappa2_store = [] # store kappa 2 values
# kappa3_store = [] # store kappa 3 values
# error_x = [] # store error in x axis
# error_y = [] # store error in y axis
i = 0
# while True:
for i in range(1000):
    start = time.time()
    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
    action = policy(tf_prev_state, ou_noise, add_noise = False) # policyde noise'i evaluate ederken 0 yap
    # Recieve state and reward from environment.
    # state, reward, done, info = env.step_1(action) # reward is -1 or 0 or 1
    state, reward, done, info = env.step_2(action[0]) # reward is -(e^2)
    x_pos.append(state[0])
    y_pos.append(state[1])
    # env.render() # uncomment for instant animation
    # buffer.record((prev_state, action, reward, state))
    # buffer.learn()
    # update_target(target_actor.variables, actor_model.variables, tau)
    # update_target(target_critic.variables, critic_model.variables, tau)

    # print(prev_state)
    print("{}th action".format(i))
    print("Goal Position",state[2:4])
    # print("Previous Error: {0}, Error: {1}, Current State: {2}".format(env.previous_error, env.error, prev_state)) # for step_1
    print("Error: {0}, Current State: {1}".format(math.sqrt(-1*reward), state)) # for step_2
    print("Action: {0},  Kappas {1}".format(action, [env.kappa1,env.kappa2,env.kappa3]))
    print("Reward is ", reward)
    print("--------------------------------------------------------------------------------")
    stop = time.time()
    env.time += (stop - start)
    error_store.append(math.sqrt(-1*reward))
    # kappa1_store.append(env.kappa1)
    # kappa2_store.append(env.kappa2)
    # kappa3_store.append(env.kappa3)
    # error_x.append(abs(state[0]-state[2]))
    # error_y.append(abs(state[1]-state[3]))
    
    # End this episode when `done` is True
    if done:
        break
    
time.sleep(1)
# %% Visualization
plt.rcParams["figure.figsize"] = (12,7)

env.visualization(x_pos,y_pos)
# plt.title(f"Initial Position is (x,y): ({initial_state[0]},{initial_state[1]}) & Target Position is (x,y): ({state[0]},{state[1]})",fontweight="bold")
plt.xlabel("Position x - [m]",fontsize=20)
plt.ylabel("Position y - [m]",fontsize=20)
plt.show()
env.close()

fig, axs = plt.subplots(2, 2,figsize=(15, 7))
fig.tight_layout(h_pad=5, w_pad=5)
axs[0, 0].plot(range(len(error_store)),error_store,c = 'red',linewidth=2,label='Total Error')
axs[0, 0].set_xlabel("Steps\n (a)",fontsize=20)
axs[0, 0].set_ylabel("Error",fontsize=20)

axs[0, 1].plot(range(len(error_x)),error_x,c = 'blue',linewidth=2, label = "X Axis")
axs[0, 1].plot(range(len(error_y)),error_y,c = 'green',linewidth=2, label = "Y Axis")
axs[0, 1].set_xlabel("Steps\n (b)",fontsize=20)
axs[0, 1].set_ylabel("Error",fontsize=20)

axs[1, 0].plot(range(len(x_pos)),x_pos,c = 'red',linewidth=2, label = "X Axis")
axs[1, 0].axhline(y=state[2])
axs[1, 0].plot(range(len(y_pos)),y_pos,c = 'green',linewidth=2, label = "Y Axis")
axs[1, 0].axhline(y=state[3])
axs[1, 0].set_xlabel("Steps\n (c)",fontsize=20)
axs[1, 0].set_ylabel("[m]",fontsize=20)

axs[1, 1].plot(range(len(kappa1_store)),kappa1_store,c = 'blue',linewidth=2, label = "Curvature-1")
axs[1, 1].plot(range(len(kappa2_store)),kappa2_store,c = 'green',linewidth=2, label = "Curvature-2")
axs[1, 1].plot(range(len(kappa3_store)),kappa3_store,c = 'red',linewidth=2, label = "Curvature-3")
axs[1, 1].set_xlabel("Steps\n (d)",fontsize=20)
axs[1, 1].set_ylabel(r"Curvature Values $\left [\frac{1}{m}  \right ]$",fontsize=20)

for ax in axs.flat:
    #ax.set_xticks(fontsize=14)
    #ax.set_yticks(fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(which='major',linewidth=0.7)
    ax.grid(which='minor',linewidth=0.5)
    ax.minorticks_on()

## Uncomment wanted plot to see the results
# # Error
# plt.plot(range(len(error_store)),error_store,c = 'red',linewidth=2)
# # plt.title("Error Plot of the Test Simulation")
# plt.xlabel("Step",fontsize=20)
# plt.ylabel("Error",fontsize=20)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.grid(which='major',linewidth=0.7)
# plt.grid(which='minor',linewidth=0.5)
# plt.minorticks_on()
# plt.show()

# # X Position
# plt.plot(range(len(x_pos)),x_pos,c = 'green',linewidth=2)
# plt.axhline(y=state[2])
# plt.grid()
# plt.legend(["Simulation Result","Reference Signal"])
# plt.title("Trajectory on the X Axis")
# plt.xlabel("Step")
# plt.ylabel("X-[m]")
# plt.show()

# # Y Position
# plt.plot(range(len(y_pos)),y_pos,c = 'red',linewidth=2)
# plt.axhline(y=state[3])
# plt.grid()
# plt.legend(["Simulation Result","Reference Signal"])
# plt.title("Trajectory on the Y Axis")
# plt.xlabel("Step")
# plt.ylabel("Y-[m]")
# plt.show()

# # X-Y Position
# plt.plot(range(len(x_pos)),x_pos,c = 'red',linewidth=2, label = "X Axis")
# plt.axhline(y=state[2])
# plt.plot(range(len(y_pos)),y_pos,c = 'green',linewidth=2, label = "Y Axis")
# plt.axhline(y=state[3])
# # plt.title("Trajectory on the X-Y Axis")
# plt.xlabel("Step",fontsize=20)
# plt.ylabel("[m]",fontsize=20)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=14)
# plt.grid(which='major',linewidth=0.7)
# plt.grid(which='minor',linewidth=0.5)
# # Show the minor ticks and grid.
# plt.minorticks_on()
# plt.show()

# # # Kappa Plots
# plt.plot(range(len(kappa1_store)),kappa1_store,c = 'blue',linewidth=2, label = "Curvature-1")
# plt.plot(range(len(kappa2_store)),kappa2_store,c = 'green',linewidth=2, label = "Curvature-2")
# plt.plot(range(len(kappa3_store)),kappa3_store,c = 'red',linewidth=2, label = "Curvature-3")
# # plt.title("Change of Curvature Values Over Time")
# plt.xlabel("Step",fontsize=20)
# plt.ylabel(r"Curvature Values $\left [\frac{1}{m}  \right ]$",fontsize=20)
# plt.legend(fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=14)
# plt.grid(which='major',linewidth=0.7)
# plt.grid(which='minor',linewidth=0.5)
# plt.minorticks_on()
# plt.show()

# # X Error
# plt.plot(range(len(error_x)),error_x,c = 'green',linewidth=2)
# plt.title("Error on the X Axis")
# plt.xlabel("Step")
# plt.ylabel("Error")
# plt.show()

# # Y Error
# plt.plot(range(len(error_y)),error_y,c = 'blue',linewidth=2)
# plt.title("Error on the Y Axis")
# plt.xlabel("Step")
# plt.ylabel("Error")
# plt.show()

# # # X-Y Error
# plt.plot(range(len(error_x)),error_x,c = 'blue',linewidth=2, label = "X Axis")
# plt.plot(range(len(error_y)),error_y,c = 'green',linewidth=2, label = "Y Axis")
# # plt.title("Error on the X-Y Axis")
# plt.xlabel("Step",fontsize=20)
# plt.ylabel("Error",fontsize=20)
# plt.legend(fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=14)
# plt.grid(which='major',linewidth=0.7)
# plt.grid(which='minor',linewidth=0.5)
# plt.minorticks_on()
# plt.show()

# %%
# theList_x = error_x
# N = 1000
# subList_x = [theList_x[n:n+N] for n in range(0, len(theList_x), N)]
# error_x_np = np.array(subList_x)
# error_x_mean = error_x_np.mean(axis=0)
# error_x_std = error_x_np.std(axis=0)

# theList_y = error_y
# N = 1000
# subList_y = [theList_y[n:n+N] for n in range(0, len(theList_y), N)]
# error_y_np = np.array(subList_y)
# error_y_mean = error_y_np.mean(axis=0)
# error_y_mean = error_y_np.std(axis=0)

# # X Error
# plt.plot(range(len(error_x_mean)),error_x_mean,c = 'blue',linewidth=3)
# plt.fill_between(range(len(error_x_mean)),error_x_mean-error_x_std,error_x_mean+error_x_std,alpha=0.2)
# plt.title("Confidence Interval of 10 episodes Mean Error on the X Axis")
# plt.xlabel("Step")
# plt.ylabel("Error")
# plt.show()

# # # Y Error
# plt.plot(range(len(error_y_mean)),error_y_mean,c = 'red',linewidth=3)
# plt.fill_between(range(len(error_y_mean)),error_y_mean-error_y_mean,error_y_mean+error_y_mean,alpha=0.2,color ="r")
# plt.title("Confidence Interval of 10 episodes Mean Error on the Y Axis")
# plt.xlabel("Step")
# plt.ylabel("Error")
# plt.show()

# # X-Y Error
# plt.plot(range(len(error_x_mean)),error_x_mean,c = 'blue',linewidth=2, label = "X Axis")
# plt.fill_between(range(len(error_x_mean)),error_x_mean-error_x_std,error_x_mean+error_x_std,alpha=0.3)
# plt.plot(range(len(error_y_mean)),error_y_mean,c = 'red',linewidth=2, label = "Y Axis")
# plt.fill_between(range(len(error_y_mean)),error_y_mean-error_y_mean,error_y_mean+error_y_mean,alpha=0.1,color ="r")
# plt.title("Confidence Interval of 10 episodes Mean Error on the X-Y Axis")
# plt.xlabel("Step")
# plt.ylabel("Error")
# plt.legend()
# plt.show()