import gym
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Subtract, Add

def NeuralNet(input_shape, output_shape):
    # Define Model
    inputs = Input(shape=input_shape)
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    value = Dense(output_shape)(x)
    advantage = Dense(output_shape)(x)
    advantage_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(advantage)
    refined_advantage = Subtract()([advantage, advantage_mean])
    outputs = Add()([value, refined_advantage])
    model = Model(inputs=inputs, outputs=outputs)
    return model

class DuelingDQN:
    def __init__(self, observation_shape, n_actions, e_greedy=0.9, gamma=0.99,
                 memory_size=10000, batch_size=32, replace_target_iter=500):
        # Hyperparameters
        self.n_features = observation_shape[0]
        self.n_actions = n_actions
        self.e_greedy = e_greedy
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.step = 0
        self.learn_step = 0
        # Memory
        self.memory = np.zeros((memory_size, self.n_features * 2 + 2))
        # Network
        self.q_eval, self.q_target = NeuralNet(observation_shape, n_actions), NeuralNet(observation_shape, n_actions)
        self.q_eval.compile(loss='mse', optimizer='adam')
        self.q_target.compile(loss='mse', optimizer='adam')

    def store_transaction(self, s, a, r, s_):
        self.memory[self.step % self.memory_size] = np.hstack((s, a, r, s_))
        self.step += 1
        
    def learn(self):
        if self.learn_step % self.replace_target_iter == 0:
            self.q_target.set_weights(self.q_eval.get_weights())
            print("updated")
        if self.step < self.memory_size:
            batch_rand_index = np.random.choice(self.step, self.batch_size)
        else:
            batch_rand_index = np.random.choice(self.memory_size, self.batch_size)
        batch_memory = self.memory[batch_rand_index]
        q_next = self.q_target.predict(batch_memory[:, -self.n_features:])
        q_target = self.q_eval.predict(batch_memory[:, :self.n_features])
        for i in range(self.batch_size):
            if abs(batch_memory[i, self.n_features+1]) != 100:
                q_target[i, batch_memory[i, self.n_features].astype(int)] = \
                    batch_memory[i, self.n_features+1] + self.gamma * q_next[i].max()
            else:
                q_target[i, batch_memory[i, self.n_features].astype(int)] = \
                    batch_memory[i, self.n_features+1]
        self.q_eval.fit(batch_memory[:, :self.n_features], q_target, epochs=10, verbose=0)
        self.learn_step += 1
        
    def choose_action(self, s):
        self.e_greedy += K.epsilon()
        if self.step < self.memory_size - 1 or np.random.uniform() > self.e_greedy:
            return np.random.choice(self.n_actions)
        return self.q_eval.predict(s.reshape(1, self.n_features)).argmax()

env = gym.make('LunarLander-v2')
agent = DuelingDQN(env.observation_space.shape, env.action_space.n)
agent.q_eval.load_weights('x.h5')

for episode in range(500):
    observation = env.reset()
    avg_r = np.zeros(100, dtype=np.float32)
    for t in range(2000):
        env.render()
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.store_transaction(observation, action, reward, observation_)
        avg_r[t % 100] = reward
        if agent.step > 500 and agent.step % 5 == 0:
            agent.learn()
        if done:
            print(f"Episode {episode} finished after {t+1} timesteps \r\t\t\t\t\t\t Reward: {avg_r.sum()}")
            break
        observation = observation_
env.close()

# Save Model (parameters)
agent.q_eval.save_weights('DuelingDQN+.h5')