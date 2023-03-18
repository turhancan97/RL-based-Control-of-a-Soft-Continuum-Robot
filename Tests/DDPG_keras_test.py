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
from continuum_robot.utils import *
from matplotlib import animation
# %matplotlib notebook
from IPython import display

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

storage = {store_name: {} for store_name in ['error', 'pos', 'kappa','reward']}
storage['error']['error_store'] = []
storage['error']['x'] = []
storage['error']['y'] = []

storage['pos']['x'] = []
storage['pos']['y'] = []

storage['kappa']['kappa1'] = []
storage['kappa']['kappa2'] = []
storage['kappa']['kappa3'] = []

storage['reward']['value'] = []
storage['reward']['effectiveness'] = []

for _ in range(5):
    env = continuumEnv()

    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(3), std_deviation=float(std_dev) * np.ones(3))

    state = env.reset() # generate random starting point for the robot and random target point.
    env.time = 0
    env.start_kappa = [env.kappa1, env.kappa2, env.kappa3] # save starting kappas
    initial_state = state[0:2]

    i = 0
    env.render_init() # uncomment for animation
    # while True:
    N = 1000
    for i in range(N):
        start = time.time()
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = policy(tf_prev_state, ou_noise, add_noise = False) # policyde noise'i evaluate ederken 0 yap
        
        # Recieve state and reward from environment.
        # state, reward, done, info = env.step_minus_euclidean_square(action[0]) # -e^2
        # state, reward, done, info = env.step_error_comparison(action[0]) # reward is -1.00 or -0.50 or 1.00
        state, reward, done, info = env.step_minus_weighted_euclidean(action[0]) # -0.7*e
        # state, reward, done, info = env.step_distance_based(action[0]) # reward is du-1 - du
        
        storage['pos']['x'].append(state[0])
        storage['pos']['y'].append(state[1])
        env.render_calculate() # uncomment for animation
        # buffer.record((prev_state, action, reward, state))
        # buffer.learn()
        # update_target(target_actor.variables, actor_model.variables, tau)
        # update_target(target_critic.variables, critic_model.variables, tau)

        # print(prev_state)
        print("{}th action".format(i))
        print("Goal Position",state[2:4])
        # print("Previous Error: {0}, Error: {1}, Current State: {2}".format(env.previous_error, env.error, prev_state)) # for step_1
        # TODO: Make it if-else to show prints below regarding to the reward
        # print("Error: {0}, Current State: {1}".format(math.sqrt(-1*reward), state)) # for step_minus_euclidean_square
        print("Error: {0}, Current State: {1}".format(env.error, state)) # for step_error_comparison
        print("Action: {0},  Kappas {1}".format(action, [env.kappa1,env.kappa2,env.kappa3]))
        print("Reward is ", reward)
        print("--------------------------------------------------------------------------------")
        stop = time.time()
        env.time += (stop - start)
        # storage['error']['error_store'].append(math.sqrt(-1*reward)) # for step_minus_euclidean_square
        storage['error']['error_store'].append((env.error)) # for step_error_comparison
        storage['kappa']['kappa1'].append(env.kappa1)
        storage['kappa']['kappa2'].append(env.kappa2)
        storage['kappa']['kappa3'].append(env.kappa3)
        storage['error']['x'].append(abs(state[0]-state[2]))
        storage['error']['y'].append(abs(state[1]-state[3]))
        storage['reward']['value'].append(reward)
        # print(env.position_dic)
        
        # End this episode when `done` is True
        if done:
            pass
            # break
    storage['reward']['effectiveness'].append(i)
                           
time.sleep(1)
print(f'{env.overshoot0} times robot tried to cross the task space')
print(f'{env.overshoot1} times random goal was generated outside of the task space')
print(f'Simulation took {(env.time)} seconds')
effectiveness_score = np.mean(storage['reward']['effectiveness'])
print(f'Average Effectiveness Score is {effectiveness_score}')

# %% Visualization of the results
env.visualization(storage['pos']['x'],storage['pos']['y'])
# plt.title(f"Initial Position is (x,y): ({initial_state[0]},{initial_state[1]}) & Target Position is (x,y): ({state[0]},{state[1]})",fontweight="bold")
plt.xlabel("Position x [m]",fontsize=15)
plt.ylabel("Position y [m]",fontsize=15)
plt.show()
env.close()
# %%
## uncomment below for animation 
# ani = env.render()
# video = ani.to_html5_video()
# html = display.HTML(video)
# display.display(html)
# plt.close()
# writergif = animation.FFMpegWriter(fps=30)
# ani.save("result.gif",writer=writergif)
# %%
# As Subplots
fig, axs = plt.subplots(2, 2)
fig.tight_layout(h_pad=5, w_pad=5)
axs[0, 0].plot(range(len(storage['error']['error_store'])),storage['error']['error_store'],c = 'red',linewidth=2,label='Total Error')
axs[0, 0].set_xlabel("Steps\n (a)",fontsize=20)
axs[0, 0].set_ylabel("Error",fontsize=20)

axs[0, 1].plot(range(len(storage['error']['x'])),storage['error']['x'],c = 'blue',linewidth=2, label = "X Axis")
axs[0, 1].plot(range(len(storage['error']['y'])),storage['error']['y'],c = 'green',linewidth=2, label = "Y Axis")
axs[0, 1].set_xlabel("Steps\n (b)",fontsize=20)
axs[0, 1].set_ylabel("Error",fontsize=20)

axs[1, 0].plot(range(len(storage['pos']['x'])),storage['pos']['x'],c = 'red',linewidth=2, label = "X Axis")
axs[1, 0].axhline(y=state[2])
axs[1, 0].plot(range(len(storage['pos']['y'])),storage['pos']['y'],c = 'green',linewidth=2, label = "Y Axis")
axs[1, 0].axhline(y=state[3])
axs[1, 0].set_xlabel("Steps\n (c)",fontsize=20)
axs[1, 0].set_ylabel("Position - [m]",fontsize=20)

axs[1, 1].plot(range(len(storage['kappa']['kappa1'])),storage['kappa']['kappa1'],c = 'blue',linewidth=2, label = "Curvature-1")
axs[1, 1].plot(range(len(storage['kappa']['kappa2'])),storage['kappa']['kappa2'],c = 'green',linewidth=2, label = "Curvature-2")
axs[1, 1].plot(range(len(storage['kappa']['kappa3'])),storage['kappa']['kappa3'],c = 'red',linewidth=2, label = "Curvature-3")
axs[1, 1].set_xlabel("Steps\n (d)",fontsize=20)
axs[1, 1].set_ylabel(r"Curvature Values $\left [\frac{1}{m}  \right ]$",fontsize=20)

for ax in axs.flat:
    #ax.set_xticks(fontsize=14)
    #ax.set_yticks(fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(which='major',linewidth=0.7)
    ax.grid(which='minor',linewidth=0.5)
    ax.minorticks_on()
# %%
## choose the plot you want to see the results
plot_choice = int(input("Please Enter 1 for Error Plot, 2 for Position Plot, 3 for Curvature Plot: "))
plot_various_results(plot_choice = plot_choice,
                     error_store = storage['error']['error_store'],
                     error_x = storage['error']['x'],
                     error_y = storage['error']['y'],
                     pos_x = storage['pos']['x'],
                     pos_y = storage['pos']['y'],
                     kappa_1 = storage['kappa']['kappa1'],
                     kappa_2 = storage['kappa']['kappa2'],
                     kappa_3 = storage['kappa']['kappa3'],
                     goal_x = state[2],
                     goal_y = state[3])

# %% Plot the average error plots
plot_average_error(error_x = storage['error']['x'], 
                   error_y = storage['error']['y'], 
                   error_store = storage['error']['error_store'], 
                   N = N)
# %% Plot Rewards
plt.plot(storage['reward']['value'],linewidth=4)
plt.xlabel("Step")
plt.ylabel("Reward x")
plt.show()