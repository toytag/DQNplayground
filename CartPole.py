import gym
import numpy as np
from tensorflow import keras

# env setup
env = gym.make('CartPole-v1')

# model setup
class DQN:
    def __init__(self, n_features, n_actions, e_greedy=0.9, 
                 memory_size=500, batch_size=32, replace_target_iter=200):
        # Hyperparameters
        self.n_features = n_features
        self.n_actions = n_actions
        self.e_greedy = e_greedy
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.step_counter = 0
        self.learn_step = 0
        # Memory
        self.memory = np.zeros((memory_size, n_features * 2 + 2))
        # Network
        self.q_eval = keras.Sequential([
            keras.layers.Dense(16, input_shape=(n_features,), 
                               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                               activation='relu'),
            keras.layers.Dense(n_actions,
                               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                               activation='linear')
        ])
        self.q_eval.compile(loss='mse', optimizer='adam')
        self.q_target = keras.Sequential([
            keras.layers.Dense(16, input_shape=(n_features,), 
                               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                               activation='relu'),
            keras.layers.Dense(n_actions,
                               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                               activation='linear')
        ])
        self.q_target.compile(loss='mse', optimizer='adam')

    def store_transaction(self, s, a, r, s_):
        self.memory[self.step_counter % self.memory_size] = np.hstack((s, [a, r], s_))
        self.step_counter += 1
        
    def learn(self):
        if self.learn_step % self.replace_target_iter == 0:
            self.q_target.set_weights(self.q_eval.get_weights())
        batch_rand_index = np.random.choice(self.memory_size, self.batch_size)
        batch_memory = self.memory[batch_rand_index]
        q_next = self.q_target.predict(batch_memory[:, -self.n_features:])
        q_eval = self.q_eval.predict(batch_memory[:, :self.n_features])
        q_target = q_eval.copy()
        q_target[np.arange(self.batch_size), batch_memory[:, self.n_features].astype(int)] = \
            batch_memory[:, self.n_features+1] + 0.9 * q_next.max(axis=1)
        self.q_eval.fit(batch_memory[:, :self.n_features], q_target, epochs=10, verbose=0)
        self.learn_step += 1
        
    def choose_action(self, s):
        if self.step_counter < self.memory_size - 1 or np.random.uniform() > self.e_greedy:
            return np.random.choice(self.n_actions)
        return self.q_eval.predict(s.reshape(1, self.n_features)).argmax()

net = DQN(env.observation_space.shape[0], env.action_space.n)

for episode in range(500):
    observation = env.reset()
    for t in range(2000):
        env.render()
        action = net.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2

        net.store_transaction(observation, action, reward, observation_)
        if net.step_counter > 200 and net.step_counter % 5 == 0:
            net.learn()
        if done:
            print(f"Episode {episode} finished after {t+1} timesteps")
            break
        observation = observation_
env.close()