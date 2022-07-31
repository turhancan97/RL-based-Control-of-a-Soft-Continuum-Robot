import sys
sys.path.append('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Reinforcement Learning')
sys.path.append('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Functions')
sys.path.append('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Keras')

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from env import continuumEnv
from DDPG import OUActionNoise, policy

env = continuumEnv()

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(3), std_deviation=float(std_dev) * np.ones(3))


# %% Evaluate

state = env.reset() # generate random starting point for the robot and random target point.
env.start_kappa = [env.kappa1, env.kappa2, env.kappa3] # save starting kappas
x_pos = []
y_pos = []
i = 0
# while True:
for i in range(750):

    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
    action = policy(tf_prev_state, ou_noise, add_noise = False) # policyde noise'i evaluate ederken 0 yap
    # Recieve state and reward from environment.
    # state, reward, done, info = env.step_1(action) # reward is -1 or 0 or 1
    state, reward, done, info = env.step_2(action[0]) # reward is -(e^2)
    x_pos.append(state[0])
    y_pos.append(state[1])
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
    
    # End this episode when `done` is True
    if done:
        break

time.sleep(2)
# %% Visualization
env.render(x_pos,y_pos)
plt.title("Trajectory of the Continuum Robot")
plt.xlabel("X - Position")
plt.ylabel("Y - Position")
plt.show()
env.close()
