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

for _ in range(1):
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
    for i in range(1000):
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
############----------------###############
## Adjust the Figure Size at the beginning ##
plt.style.use('ggplot') # ggplot sytle plots
plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams["axes.titlesize"] = 'xx-large'
plt.rcParams["axes.labelsize"] = 'xx-large'
plt.rcParams['animation.ffmpeg_path'] = '/home/tkargin/miniconda3/envs/continuum-rl/bin/ffmpeg' 
## plt.rcParams.keys() ## To see the plot adjustment parameters
############----------------###############
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
## Uncomment wanted plot to see the results
# # Error
# plt.plot(range(len(storage['error']['error_store'])),storage['error']['error_store'],c = 'red',linewidth=2,label='Total Error')
# plt.title("Error Plot of the Test Simulation")
# plt.xlabel("Step",fontsize=20)
# plt.ylabel("Error",fontsize=20)
# plt.legend(fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.grid(which='major',linewidth=0.7)
# plt.grid(which='minor',linewidth=0.5)
# plt.minorticks_on()
# plt.show()

# # X Position
# plt.plot(range(len(storage['pos']['x'])),storage['pos']['x'],c = 'green',linewidth=2)
# plt.axhline(y=state[2])
# plt.grid()
# plt.legend(["Simulation Result","Reference Signal"])
# plt.title("Trajectory on the X Axis")
# plt.xlabel("Step")
# plt.ylabel("X-[m]")
# plt.show()

# # Y Position
# plt.plot(range(len(storage['pos']['y'])),storage['pos']['y'],c = 'red',linewidth=2)
# plt.axhline(y=state[3])
# plt.grid()
# plt.legend(["Simulation Result","Reference Signal"])
# plt.title("Trajectory on the Y Axis")
# plt.xlabel("Step")
# plt.ylabel("Y-[m]")
# plt.show()

# # X-Y Position
# plt.plot(range(len(storage['pos']['x'])),storage['pos']['x'],c = 'red',linewidth=2, label = "X Axis")
# plt.axhline(y=state[2])
# plt.plot(range(len(storage['pos']['y'])),storage['pos']['y'],c = 'green',linewidth=2, label = "Y Axis")
# plt.axhline(y=state[3],label='Reference Signal')
# plt.title("Trajectory on the X-Y Axis")
# plt.xlabel("Step",fontsize=20)
# plt.ylabel("Position [m]",fontsize=20)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(fontsize=15)
# plt.grid(which='major',linewidth=0.7)
# plt.grid(which='minor',linewidth=0.5)
# # Show the minor ticks and grid.
# plt.minorticks_on()
# plt.show()

# # Kappa Plots
# plt.plot(range(len(storage['kappa']['kappa1'])),storage['kappa']['kappa1'],c = 'blue',linewidth=2, label = "Curvature-1")
# plt.plot(range(len(storage['kappa']['kappa2'])),storage['kappa']['kappa2'],c = 'green',linewidth=2, label = "Curvature-2")
# plt.plot(range(len(storage['kappa']['kappa3'])),storage['kappa']['kappa3'],c = 'red',linewidth=2, label = "Curvature-3")
# plt.title("Change of Curvature Values Over Time")
# plt.xlabel("Step",fontsize=20)
# plt.ylabel(r"Curvature Values $\left [\frac{1}{m}  \right ]$",fontsize=18)
# plt.legend(fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.grid(which='major',linewidth=0.7)
# plt.grid(which='minor',linewidth=0.5)
# plt.minorticks_on()
# plt.show()

# # X Error
# plt.plot(range(len(storage['error']['x'])),storage['error']['x'],c = 'green',linewidth=2)
# plt.title("Error on the X Axis")
# plt.xlabel("Step")
# plt.ylabel("Error")
# plt.show()

# # Y Error
# plt.plot(range(len(storage['error']['y'])),storage['error']['y'],c = 'blue',linewidth=2)
# plt.title("Error on the Y Axis")
# plt.xlabel("Step")
# plt.ylabel("Error")
# plt.show()

# # X-Y Error
# plt.plot(range(len(storage['error']['x'])),storage['error']['x'],c = 'blue',linewidth=2, label = "X Axis")
# plt.plot(range(len(storage['error']['y'])),storage['error']['y'],c = 'green',linewidth=2, label = "Y Axis")
# plt.title("Error on the X-Y Axis")
# plt.xlabel("Step",fontsize=20)
# plt.ylabel("Error",fontsize=20)
# plt.legend(fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.grid(which='major',linewidth=0.7)
# plt.grid(which='minor',linewidth=0.5)
# plt.minorticks_on()
# plt.show()

# %%
N = 1000

theList_x = storage['error']['x']
subList_x = [theList_x[n:n+N] for n in range(0, len(theList_x), N)]
error_x_np = np.array(subList_x)
error_x_mean = error_x_np.mean(axis=0)
error_x_std = error_x_np.std(axis=0)

theList_y = storage['error']['y']
subList_y = [theList_y[n:n+N] for n in range(0, len(theList_y), N)]
error_y_np = np.array(subList_y)
error_y_mean = error_y_np.mean(axis=0)
error_y_std = error_y_np.std(axis=0)

theList_combined = storage['error']['error_store']
subList_combined = [theList_combined[n:n+N] for n in range(0, len(theList_combined), N)]
error_combined_np = np.array(subList_combined)
error_combined_mean = error_combined_np.mean(axis=0)
error_combined_std = error_combined_np.std(axis=0)

# X Error
plt.plot(range(len(error_x_mean)),error_x_mean,c = 'blue',linewidth=3)
plt.fill_between(range(len(error_x_mean)),error_x_mean-error_x_std,error_x_mean+error_x_std,alpha=0.2,color='b')
plt.title("10 episodes of Distance Error on the X Axis with Confidence Band")
plt.xlabel("Step")
plt.ylabel("Error")
plt.show()

# # Y Error
plt.plot(range(len(error_y_mean)),error_y_mean,c = 'red',linewidth=3)
plt.fill_between(range(len(error_y_mean)),error_y_mean-error_y_std,error_y_mean+error_y_std,alpha=0.2,color ="r")
plt.title("10 episodes of Distance Error on the Y Axis with Confidence Band")
plt.xlabel("Step")
plt.ylabel("Error")
plt.show()

# X-Y Error
plt.plot(range(len(error_x_mean)),error_x_mean,c = 'blue',linewidth=2, label = "X Axis")
plt.fill_between(range(len(error_x_mean)),error_x_mean-error_x_std,error_x_mean+error_x_std,alpha=0.3,color='b')
plt.plot(range(len(error_y_mean)),error_y_mean,c = 'red',linewidth=2, label = "Y Axis")
plt.fill_between(range(len(error_y_mean)),error_y_mean-error_y_std,error_y_mean+error_y_std,alpha=0.1,color ="r")
plt.title("10 episodes of Distance Error on the X-Y Axis with Confidence Band")
plt.xlabel("Step")
plt.ylabel("Error")
plt.legend()
plt.show()

# Combined Error
plt.plot(range(len(error_combined_mean)),error_combined_mean,c = 'black',linewidth=3)
plt.fill_between(range(len(error_combined_mean)),error_combined_mean-error_combined_std,error_combined_mean+error_combined_std,alpha=0.2,color ="k")
plt.title("10 episodes of Total Distance Error with Confidence Band")
plt.xlabel("Step")
plt.ylabel("Error")
plt.show()
# %% Plot Rewards
plt.plot(storage['reward']['value'],linewidth=4)
plt.xlabel("Step")
plt.ylabel("Reward 4")
plt.show()
# %%
