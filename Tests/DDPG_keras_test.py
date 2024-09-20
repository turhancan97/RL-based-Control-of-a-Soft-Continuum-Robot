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
from DDPG import OUActionNoise, policy, config
plt.style.use('../continuum_robot/plot.mplstyle')
from continuum_robot.utils import *
from matplotlib import animation
# %matplotlib notebook
from IPython import display

# %% Evaluate the policy

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

episode_number = 1
for _ in range(episode_number):
    env = continuumEnv() # initialize environment

    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(3), std_deviation=float(std_dev) * np.ones(3))

    state = env.reset() # generate random starting point for the robot and random target point.
    env.time = 0
    env.start_kappa = [env.kappa1, env.kappa2, env.kappa3] # save starting kappas
    initial_state = state[0:2]

    env.render_init() # uncomment for animation

    N = 500
    step = 0
    for step in range(N): # or while True:
        start = time.time()
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = policy(tf_prev_state, ou_noise, add_noise = False) # policyde noise'i evaluate ederken 0 yap

        # Recieve state and reward from environment.
        # 'step_minus_euclidean_square' is e^2
        # 'step_minus_weighted_euclidean' is 0.7*e
        # 'step_error_comparison' is -1.00 or -0.50 or 1.00
        # 'step_distance_based' is du-1 - du

        state, reward, done, info = env.step(action[0], reward_function = config['reward']['function'])
        
        storage['pos']['x'].append(state[0])
        storage['pos']['y'].append(state[1])
        env.render_calculate() # uncomment for animation

        print("{}th action".format(step))
        print("Goal Position",state[2:4])
        if config['reward']['function'] == 'step_minus_euclidean_square':
            print("Error: {0}, Current State: {1}".format(math.sqrt(-1*reward), state)) # for step_minus_euclidean_square
        else:
            print("Error: {0}, Current State: {1}".format(env.error, state)) # for other rewards
        print("Action: {0},  Kappas {1}".format(action, [env.kappa1,env.kappa2,env.kappa3]))
        print("Reward is ", reward)
        print("--------------------------------------------------------------------------------")
        stop = time.time()
        env.time += (stop - start)
        if config['reward']['function'] == 'step_minus_euclidean_square':
            storage['error']['error_store'].append(math.sqrt(-1*reward)) # for step_minus_euclidean_square
        else:
            storage['error']['error_store'].append((env.error)) # for other rewards
        storage['kappa']['kappa1'].append(env.kappa1)
        storage['kappa']['kappa2'].append(env.kappa2)
        storage['kappa']['kappa3'].append(env.kappa3)
        storage['error']['x'].append(abs(state[0]-state[2]))
        storage['error']['y'].append(abs(state[1]-state[3]))
        storage['reward']['value'].append(reward)
        # print(env.position_dic)
        
        # End this episode when `done` is True
        if done:
            # pass
            break
    storage['reward']['effectiveness'].append(step)
                           
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
# # uncomment below for animation 
# ani = env.render()
# video = ani.to_html5_video()
# html = display.HTML(video)
# display.display(html)
# plt.close()
# writergif = animation.FFMpegWriter(fps=30)
# ani.save("../docs/videos/result.gif",writer=writergif)
# %%
# As Subplots
sub_plot_various_results(error_store = storage['error']['error_store'],
                         error_x = storage['error']['x'],
                         error_y = storage['error']['y'],
                         pos_x = storage['pos']['x'],
                         pos_y = storage['pos']['y'],
                         kappa_1 = storage['kappa']['kappa1'],
                         kappa_2 = storage['kappa']['kappa2'],
                         kappa_3 = storage['kappa']['kappa3'],
                         goal_x = state[2],
                         goal_y = state[3])
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
                   N = N, 
                   episode_number=episode_number)
# %% Plot Rewards
plt.plot(storage['reward']['value'],linewidth=4)
plt.xlabel("Step")
plt.ylabel(f"Reward {config['reward']['function']}")
plt.show()
