import sys
sys.path.append('../')
sys.path.append('../Reinforcement Learning')
sys.path.append('../Tests')

import os
import yaml
# import gym
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras import layers
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import math
# from forward_velocity_kinematics import three_section_planar_robot
from env import continuumEnv


# * Read the config file
# Get the absolute path to the directory containing this script
dir_path = os.path.dirname(os.path.realpath(__file__))
# Construct the absolute path to the file
file_path = os.path.join(dir_path, "config.yaml")
# load config file
with open(file_path, "r") as file:
    config = yaml.safe_load(file)

env = continuumEnv()

num_states = env.observation_space.shape[0] * 2 # multiply by 2 because we have also goal state
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))
start_time = time.time()
# %%
class OUActionNoise:
    """
    It creates a noise process that is correlated with the previous noise value 
    # TODO: add nice description for the noise.
    """
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
            

class Buffer:
    """
    Class docstrings go here. # TODO: add nice description for the Buffer.
    """
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1][0]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        # ===================================================================== #
		#                               Actor Model                             #
        # ===================================================================== #
        #                               Critic Model                            #
        # ===================================================================== #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=TRAIN)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=TRAIN
            )
            critic_value = critic_model([state_batch, action_batch], training=TRAIN)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=TRAIN)
            critic_value = critic_model([state_batch, actions], training=TRAIN)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

    # ========================================================================= #
	#                              Model Definitions                            #
	# ========================================================================= #
    
def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003) # minval=-0.003, maxval=0.003

    inputs = layers.Input(shape=(num_states,))
    # inputs = layers.Dropout(0.2)(inputs) # delete
    # inputs = layers.BatchNormalization()(inputs)  # delete
    out = layers.Dense(512, activation="relu")(inputs) # 512
    # out = layers.BatchNormalization()(out)  # delete
    out = layers.Dense(256, activation="relu")(out) # 256
    # out = layers.BatchNormalization()(out)  # delete
    out = layers.Dense(128, activation="relu")(out) # 256
    # out = layers.BatchNormalization()(out)  # delete
    # out = layers.Dense(512, activation="relu")(out) # 512
    # out = layers.BatchNormalization()(out) # delete
    # out = layers.Dense(256, activation="relu")(out) # delete
    
    # Outputs the actions - We have 3 actions
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 1.0 for Continuum Robot (Kappa dot).
    outputs = outputs * upper_bound # * env.dt
    # outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    # state_input = layers.Dropout(0.2)(state_input) # delete
    # state_input = layers.BatchNormalization()(state_input) # delete
    state_out = layers.Dense(64, activation="relu")(state_input) # 32
    # state_out = layers.BatchNormalization()(state_out) # delete
    state_out = layers.Dense(32, activation="relu")(state_out) # 64
    # state_out = layers.BatchNormalization()(state_out) # delete
    state_out = layers.Dense(32, activation="relu")(state_out) # 128

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    # action_input = layers.Dropout(0.2)(action_input) # delete
    # action_input = layers.BatchNormalization()(action_input) # delete
    action_out = layers.Dense(32, activation="relu")(action_input) # 128
    # action_out = layers.BatchNormalization()(action_out) # delete
    # action_out = layers.Dense(64, activation="relu")(action_out) # 64
    # action_out = layers.BatchNormalization()(action_out) # delete
    # action_out = layers.Dense(32, activation="relu")(action_out) # 32
    
    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat) # 256
    # out = layers.BatchNormalization()(out) # delete
    out = layers.Dense(256, activation="relu")(out) # 256
    # out = layers.BatchNormalization()(out)  # delete
    # out = layers.Dense(128, activation="relu")(out) # 128
    # out = layers.BatchNormalization()(out) # delete
    # out = layers.Dense(256, activation="relu")(out) # delete
    # out = layers.BatchNormalization()(out)  # delete
    outputs = layers.Dense(1)(out) # outputs the Q values (layers.Dense(1)(out))

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object,add_noise=True):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object() # change to 0 to delete noise
    # Adding noise to action
    if add_noise:
        sampled_actions = sampled_actions.numpy() + noise
    
    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

# %%

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# actor_model.load_weights("continuum_actor.h5")
# critic_model.load_weights("continuum_critic.h5")
# target_actor.load_weights("continuum_target_actor.h5")
# target_critic.load_weights("continuum_target_critic.h5")

# Learning rate for actor-critic models
critic_lr = 1e-3        # learning rate of the critic
actor_lr = 1e-4         # learning rate of the actor

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 400
# Discount factor for future rewards
gamma = 0.99            # discount factor
# Used to update target networks
tau = 5e-3              # for soft update of target parameters

buffer = Buffer(int(5e5), 128) # Buffer(50000, 64)

# %% Train or Evaluate
# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
counter = 0
avg_reward = 0

TRAIN = False

if TRAIN:
    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(std_dev) * np.ones(num_actions))
    
    for ep in range(total_episodes):
        
        # prev_state = env.reset_known() # starting position is always same
        prev_state = env.reset() # starting postion is random (within task space)
        # high = np.array([0.18, 0.3], dtype=np.float32)
        # low = np.array([-0.25, -0.1], dtype=np.float32)
        # env.q_goal = np.random.uniform(low=low, high=high)
        if ep % 50 == 0:
            print('Episode Number',ep)
            print("Initial Position is",prev_state[0:2])
            print("===============================================================")
            print("Target Position is",prev_state[2:4])
            print("===============================================================")
            print("Initial Kappas are ",[env.kappa1,env.kappa2,env.kappa3])
            print("===============================================================")
            print("Goal Kappas are ",[env.target_k1,env.target_k2,env.target_k3])
            print("===============================================================")
        
        # time.sleep(2) # uncomment when training in local computer
        episodic_reward = 0
    
        # while True:
        for i in range(500):
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()
    
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = policy(tf_prev_state, ou_noise)
    
            # Recieve state and reward from environment.
            # 'step_minus_euclidean_square' is e^2
            # 'step_minus_weighted_euclidean' is 0.7*e
            # 'step_error_comparison' is -1.00 or -0.50 or 1.00
            # 'step_distance_based' is du-1 - du
            state, reward, done, info = env.step(action[0], reward_function = config['reward']['function'])
            
            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward
    
            buffer.learn()
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)
    
            # End this episode when `done` is True
            if done:
                counter += 1
                break
    
            prev_state = state
            # print(prev_state)
            # # Uncomment below when training in local computer
            # print("Episode Number {0} and {1}th action".format(ep,i))
            # print("Goal Position",prev_state[2:4])
            # # print("Previous Error: {0}, Error: {1}, Current State: {2}".format(env.previous_error, env.error, prev_state)) # for step_1
            # print("Error: {0}, Current State: {1}".format(math.sqrt(-1*reward), prev_state)) # for step_2
            # print("Action: {0},  Kappas {1}".format(action, [env.kappa1,env.kappa2,env.kappa3]))
            # print("Reward is ", reward)
            # print("{0} times robot reached to the target".format(counter))
            # print("Avg Reward is {0}, Episodic Reward is {1}".format(avg_reward,episodic_reward))
            # print("--------------------------------------------------------------------------------")
    
        ep_reward_list.append(episodic_reward)
    
        # Mean of 250 episodes
        avg_reward = np.mean(ep_reward_list[-100:])
        if ep % 1 == 0:
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            time.sleep(0.5)
        avg_reward_list.append(avg_reward)
    
    print(f'{counter} times robot reached the target point in total {total_episodes} episodes')
    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(avg_reward_list)+1), avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")

    with open('avg_reward_list.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(avg_reward_list, f, pickle.HIGHEST_PROTOCOL)
    
    # Episodes versus Rewards
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(ep_reward_list)+1), ep_reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.show()

    with open('ep_reward_list.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(ep_reward_list, f, pickle.HIGHEST_PROTOCOL)
    
    # Save Weights
    actor_model.save_weights("experiment/continuum_actor.h5")
    critic_model.save_weights("experiment/continuum_critic.h5")
    target_actor.save_weights("experiment/continuum_target_actor.h5")
    target_critic.save_weights("experiment/continuum_target_critic.h5")
    end_time = time.time() - start_time
    print('Total Overshoot 0: ', env.overshoot0)
    print('Total Overshoot 1: ', env.overshoot1)
    print(f'Total Elapsed Time is {int(end_time)/60} minutes')
else:
    actor_model.load_weights(f"../Keras/{config['goal_type']}/{config['reward']['file']}/model/continuum_actor.h5")
    critic_model.load_weights(f"../Keras/{config['goal_type']}/{config['reward']['file']}/model/continuum_critic.h5")
    target_actor.load_weights(f"../Keras/{config['goal_type']}/{config['reward']['file']}/model/continuum_target_actor.h5")
    target_critic.load_weights(f"../Keras/{config['goal_type']}/{config['reward']['file']}/model/continuum_target_critic.h5")
    
    # state = env.reset() # generate random starting point for the robot and random target point.
    # env.start_kappa = [env.kappa1, env.kappa2, env.kappa3] # save starting kappas
    # x_pos = []
    # y_pos = []
    # i = 0
    # # while True:
    # for i in range(750):
    
    #     tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
    #     action = policy(tf_prev_state, ou_noise, add_noise = False) # policyde noise'i evaluate ederken 0 yap
    #     # Recieve state and reward from environment.
    #     # state, reward, done, info = env.step_1(action) # reward is -1 or 0 or 1
    #     state, reward, done, info = env.step_2(action[0]) # reward is -(e^2)
    #     x_pos.append(state[0])
    #     y_pos.append(state[1])
    #     # buffer.record((prev_state, action, reward, state))
    #     # buffer.learn()
    #     # update_target(target_actor.variables, actor_model.variables, tau)
    #     # update_target(target_critic.variables, critic_model.variables, tau)
    
    #     # print(prev_state)
    #     print("{}th action".format(i))
    #     print("Goal Position",state[2:4])
    #     # print("Previous Error: {0}, Error: {1}, Current State: {2}".format(env.previous_error, env.error, prev_state)) # for step_1
    #     print("Error: {0}, Current State: {1}".format(math.sqrt(-1*reward), state)) # for step_2
    #     print("Action: {0},  Kappas {1}".format(action, [env.kappa1,env.kappa2,env.kappa3]))
    #     print("Reward is ", reward)
    #     print("--------------------------------------------------------------------------------")
        
    #     # End this episode when `done` is True
    #     if done:
    #         break
    
    # time.sleep(2)
    # # Visualization
    # env.render(x_pos,y_pos)
    # plt.title("Trajectory of the Continuum Robot")
    # plt.xlabel("X - Position")
    # plt.ylabel("Y - Position")
    # plt.show()
    # env.close()