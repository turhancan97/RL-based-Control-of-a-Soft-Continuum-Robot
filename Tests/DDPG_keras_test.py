import sys
sys.path.append('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Reinforcement Learning')
sys.path.append('C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Functions')

# import gym
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from forward_velocity_kinematics import trans_mat_cc, coupletransformations
from env import continuumEnv

env = continuumEnv()

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

class OUActionNoise:
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
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
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
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    
    # Outputs the actions - We have 3 actions
    outputs = layers.Dense(3, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 1.0 for Continuum Robot (Kappa dot).
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out) # outputs the Q values

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = 0 # noise_object() # uncomment to add noise
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

std_dev = 0.1
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(3))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights
actor_model.load_weights("C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Keras/case1_known_starting_and_target_point/continuum_actor.h5")
critic_model.load_weights("C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Keras/case1_known_starting_and_target_point/continuum_critic.h5")
target_actor.load_weights("C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Keras/case1_known_starting_and_target_point/continuum_target_actor.h5")
target_critic.load_weights("C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Keras/case1_known_starting_and_target_point/continuum_target_critic.h5")


# Learning rate for actor-critic models
critic_lr = 0.001 # 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 1
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.01 # 0.005

buffer = Buffer(50000, 64)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []


t = time.time()
# Takes about 4 min to train
for ep in range(total_episodes):

    prev_state = env.reset_known() # starting position is always same
    # prev_state = env.reset_unknown() # starting postion is random (within task space)
    start_kappa = [env.kappa1, env.kappa2, env.kappa3]
    print(prev_state)
    time.sleep(1)
    episodic_reward = 0
    x_pos = []
    y_pos = []

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise)
        # print(action)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)
        x_pos.append(state[0])
        y_pos.append(state[1])
        # print(env.kappa1,env.kappa2,env.kappa3)
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state
        # print(prev_state)
        print("Episode number ", ep)
        print("Error: {0}, Current State: {1}".format(math.sqrt(-1*reward), prev_state))
        print("Action: {0},  Kappas {1}".format(action, [env.kappa1,env.kappa2,env.kappa3]))
        print("--------------------------------------------------------------------------------")

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    time.sleep(1)
    avg_reward_list.append(avg_reward)

elapsed = time.time() - t
print("Elapsed Time is",elapsed)
# actor_model.save_weights("C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Keras/case1_known_starting_and_target_point/continuum_actor.h5")
# critic_model.save_weights("C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Keras/case1_known_starting_and_target_point/continuum_critic.h5")
# target_actor.save_weights("C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Keras/case1_known_starting_and_target_point/continuum_target_actor.h5")
# target_critic.save_weights("C:/Users/Asus/Desktop/Master-Lectures/3rd Semester/Thesis/Githubs/my_project/Thesis-Project/RL-based-Control-of-a-Soft-Continuum-Robot/Keras/case1_known_starting_and_target_point/continuum_target_critic.h5")
# %% Visualization
T1_cc = trans_mat_cc(start_kappa[0],env.l[0])
T1_tip = np.reshape(T1_cc[len(T1_cc)-1,:],(4,4),order='F');
# section 2
T2 = trans_mat_cc(start_kappa[1],env.l[1]);
T2_cc = coupletransformations(T2,T1_tip);
T2_tip = np.reshape(T2_cc[len(T2_cc)-1,:],(4,4),order='F');
# section 3
T3 = trans_mat_cc(start_kappa[2],env.l[2]);
T3_cc = coupletransformations(T3,T2_tip);
# Plot the trunk with three sections and point the section seperation

plt.plot([-0.025, 0.025],[0,0],'black',linewidth=5)
plt.plot(T1_cc[:,12],T1_cc[:,13],'b',linewidth=3)
#plt.scatter(T1_cc[-1,12],T1_cc[-1,13],linewidths=5,color = 'black')
plt.plot(T2_cc[:,12],T2_cc[:,13],'r',linewidth=3)
#plt.scatter(T2_cc[-1,12],T2_cc[-1,13],linewidths=5,color = 'black')
plt.plot(T3_cc[:,12],T3_cc[:,13],'g',linewidth=3)
plt.scatter(T3_cc[-1,12],T3_cc[-1,13],linewidths=5,color = 'black')

T1_cc = trans_mat_cc(env.kappa1,env.l[0])
T1_tip = np.reshape(T1_cc[len(T1_cc)-1,:],(4,4),order='F');
# section 2
T2 = trans_mat_cc(env.kappa2,env.l[1]);
T2_cc = coupletransformations(T2,T1_tip);
T2_tip = np.reshape(T2_cc[len(T2_cc)-1,:],(4,4),order='F');
# section 3
T3 = trans_mat_cc(env.kappa3,env.l[2]);
T3_cc = coupletransformations(T3,T2_tip);

# Plot the trunk with three sections and point the section seperation
plt.plot(T1_cc[:,12],T1_cc[:,13],'b',linewidth=3)
#plt.scatter(T1_cc[-1,12],T1_cc[-1,13],linewidths=5,color = 'black')
plt.plot(T2_cc[:,12],T2_cc[:,13],'r',linewidth=3)
#plt.scatter(T2_cc[-1,12],T2_cc[-1,13],linewidths=5,color = 'black')
plt.plot(T3_cc[:,12],T3_cc[:,13],'g',linewidth=3)
plt.scatter(T3_cc[-1,12],T3_cc[-1,13],linewidths=5,color = 'black')

env.render2(x_pos,y_pos)
plt.title("Trajectory of the Continuum Robot")
plt.xlabel("X - Position")
plt.ylabel("Y - Position")
plt.show()