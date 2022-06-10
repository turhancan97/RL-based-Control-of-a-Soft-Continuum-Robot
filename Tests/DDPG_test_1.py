import sys
sys.path.append('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Reinforcement Learning')

# import gym
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import time
from env import continuumEnv

env = continuumEnv()

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(3, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model

def policy(state):
    sampled_actions = tf.squeeze(actor_model(state))
    # Adding noise to action
    sampled_actions = sampled_actions.numpy()

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


actor_model = get_actor()
actor_model.load_weights("C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Keras/case1_known_starting_and_target_point/continuum_target_actor.h5")
#actor_model.get_weights()

# Showing the video
t = time.time()
for _ in range(10):
    observation = env.reset()
    done = False
    total_reward = 0
    x_pos = []
    y_pos = []
    while not done:
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(observation), 0)
        action = policy(tf_prev_state)
        observation, reward, done, info = env.step(action)
        print(observation)
        x_pos.append(observation[0])
        y_pos.append(observation[1])
        # env.render()
        # plt.pause(0.001)
        total_reward = total_reward + reward
    print('total reward:' + '%.2f' % total_reward)
    time.sleep(1)
    
# env.render(x_pos,y_pos)
elapsed = time.time() - t
print(elapsed)
env.render2(x_pos,y_pos)
plt.legend(["Target Position","Motion"])
plt.title("2D Motion of Tip of the Continuum Robot")
plt.xlabel("X - Position")
plt.ylabel("Y - Position")
plt.show()