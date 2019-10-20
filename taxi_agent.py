import gym
import gym.spaces
import numpy as np
import random
from collections import deque, defaultdict
import math
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from random import shuffle


class Agent():
    # Agent with a neural network
    def __init__(self, obs_space, action_space):
        self.obs_space = obs_space
        self.action_space = action_space
        self.memory = deque(maxlen=15000)
        self.gamma = 0.95
        self.batch_size = 32
        self.model = self._build_model()

    def remember(self, observation, action, reward, next_observation):
        if len(self.memory) > self.memory.maxlen:
            if np.random.random() < 0.5:
                shuffle(self.memory)
            self.memory.popleft()
        self.memory.append((observation, action, reward, next_observation))

    def get_q(self, observation):
        np_obs = np.reshape(observation, [-1, self.obs_space])
        return self.model.predict(np_obs)

    def _build_model(self):
        # Builds really simple neural network without hidden layer or bias
        model = Sequential()
        model.add(Dense(self.action_space, input_shape=(self.obs_space,), activation='linear', use_bias=False))
        model.compile(optimizer=Adam(), loss='mse')
        return model

    def update_action(self):
        sample_transitions = random.sample(self.memory, self.batch_size)
        random.shuffle(sample_transitions)
        batch_observations = []
        batch_targets = []

        for old_observation, action, reward, observation in sample_transitions:
            # Reshape targets to output dimension(=2)
            targets = np.reshape(
                self.get_q(old_observation),
                self.action_space)
            targets[action] = reward  # Set Target Value
            if observation is not None:
                # If the old state is not a final state, also consider the
                # discounted future reward
                predictions = self.get_q(observation)
                new_action = np.argmax(predictions)
                targets[action] += self.gamma * predictions[0, new_action]

            # Add Old State to observations batch
            batch_observations.append(old_observation)
            batch_targets.append(targets)  # Add target to targets batch

        # Update the model using Observations and their corresponding Targets
        np_obs = np.reshape(batch_observations, [-1, self.obs_space])
        np_targets = np.reshape(batch_targets, [-1, self.action_space])
        self.model.fit(np_obs, np_targets, epochs=1, verbose=0)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)


class TableAgent():
    # Normal Q-learning agent that uses a table
    def __init__(self, actions, states):
        self.actions = actions
        self.states = states
        self.q_table = np.full((states, actions), 150) # Set every value to 150 initially to guarentee exploring
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9972
        self.alpha = 0.8
        self.gamma = 0.95

    def select_action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        self.q_table[state][action] = self.q_table[state][action] + self.alpha * \
            (reward + (self.gamma * np.max(self.q_table[next_state])) - self.q_table[state][action]) 


def train_table():
    # Trains a Q-learning agent and returns the Q-table
    env = gym.make('Taxi-v3')
    agent = TableAgent(env.action_space.n, env.observation_space.n)

    num_episodes = 10000
    avg_rewards = deque(maxlen=num_episodes)
    best_avg_reward = -math.inf
    recent_rewards = deque(maxlen=100)

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            if done:
                recent_rewards.append(episode_reward)
                break

        if (i_episode >= 100):
            avg_reward = np.mean(recent_rewards)
            avg_rewards.append(avg_reward)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon = max(
                agent.epsilon_min,
                agent.epsilon *
                agent.epsilon_decay)

        print("\rEpisode {:05d}, epsilon = {:.3f}, best average = {:.3f}".format(
            i_episode, agent.epsilon, best_avg_reward), end="")

        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break

    return agent.q_table


def create_agent_weights(q_table):
    # Building model (agent)
    agent = Agent(500, 6)
    
    # Creating weights
    weights = np.zeros((1, 500, 6))
    best_actions = np.argmax(q_table, axis=1)
    for n in range(0, 500):
        weights[0][n][best_actions[n]] = 1

    # Setting and saving the weights
    agent.model.set_weights(weights)
    agent.model.save_weights('models/taxi_model.h5')

    return weights


def train_agent_weights(q_table):
    # This function trains the model of an agent on the Q-table
    # Parameters
    epochs = 15000
    batch_size = 16

    # Building model/agent
    agent = Agent(500, 6)

    # Creating data
    x_train = np.array([decode(i) for i in range(0,500)])
    best_actions = np.argmax(q_table, axis=1)
    y_train = np.zeros((500,6))
    for n in range(500):
        y_train[n][best_actions[n]]= 1

    # Train the model
    agent.model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1
    )

    # Saving the model
    agent.model.save_weights('models/taxi_model.h5')


def train():
    # Trains the Agent the same way the CartPole agent was trained
    # This function rarely converges
    env = gym.make('Taxi-v3')
    action_space = env.action_space.n
    observation_space = env.observation_space.n
    agent = Agent(observation_space, action_space)

    episodes = 50000
    max_steps = 1000
    epsilon = 1
    epsilon_decay = 0.9999
    epsilon_min = 0.1
    scores = []  # A list of all game scores
    recent_scores = []  # List that hold most recent 100 game scores

    for episode in range(episodes):
        observation = env.reset()
        observation = decode(observation)
        episode_reward = 0
        for step in range(max_steps):
            old_observation = observation
            if np.random.random() < epsilon:
                # Take random action (explore)
                action = np.random.choice(range(action_space))
            else:
                # Query the model and get Q-values for possible actions
                q_values = agent.get_q(observation)
                action = np.argmax(q_values)
            # Take the selected action and observe next state
            observation, reward, done, _ = env.step(action)
            observation = decode(observation)
            episode_reward += reward
            if done or step == max_steps - 1:
                scores.append(episode_reward)  # Append final score
                # accelerate learning
                if reward == 20:
                    reward += 200
                else:
                    reward += -30
                # Calculate recent scores
                if len(scores) > 100:
                    recent_scores = scores[-100:]
                # Add the abservation to replay memory
                agent.remember(old_observation, action, reward, None)
                break
            # Add the abservation to replay memory
            agent.remember(old_observation, action, reward, observation)
            # Update the Deep Q-Network Model (only with a chance of 25% and when the last score was worse than 9.7)
            if np.random.random() < 0.3 and len(scores) > 0 and len(agent.memory) >= agent.batch_size:
                if scores[-1] < 9.7:
                    agent.update_action()

        print("Episode {:03d} , epsilon = {:.4f}, score = {:03d}".format(
                episode,
                epsilon,
                episode_reward))
        # If mean over the last 100 Games is >9.7, then success
        if np.mean(recent_scores) > 9.7:
            print("\nEnvironment solved in {} episodes.".format(episode), end="")
            break
        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        # Periodically saving the model
        if episode % 500 == 0:
            agent.model.save('models/taxi_model_{}.h5'.format(episode))

    # Saving the model
    agent.model.save('models/taxi_model.h5')

    # Plotting of the results
    plt.plot(scores)
    plt.title('Training Phase')
    plt.ylabel('Time Steps')
    plt.xlabel('Trial')
    plt.savefig('results/TaxiAgentTraining.png', bbox_inches='tight')
    plt.show()


def test_agent():
    env = gym.make("Taxi-v3")
    action_space = env.action_space.n
    observation_space = env.observation_space.n
    agent = Agent(observation_space, action_space)
    agent.load("models/taxi_model.h5")
    scores = []

    for _ in range(100):
        obs = env.reset()
        obs = decode(obs)
        episode_reward = 0
        while True:
            q_values = agent.get_q(obs)
            action = np.argmax(q_values)
            obs, reward, done, _ = env.step(action)
            obs = decode(obs)
            episode_reward += reward
            if done:
                break
        scores.append(episode_reward)

    # Printing of the results
    print("Scores: ", scores)
    print("Mean: ", np.mean(scores))

    # Plotting of the results
    plt.plot(scores)
    plt.title('Testing Phase')
    plt.ylabel('Time Steps')
    plt.ylim(ymax=10)
    plt.xlabel('Trial')
    plt.savefig('results/TaxiAgentTesting.png', bbox_inches='tight')
    plt.show()


def decode(i):
    # Array of length 500 as one-hot vector
    state = np.zeros(500)
    state[i] = 1
    return state


def decode_to_18bit(i):
    # This function decodes the state to an array of length 18 containing the
    # taxi location, passenger location and destination
    out = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5
    state = np.zeros(18)
    # 5 bits for the taxi row
    state[out[3]] = 1
    # 5 bits for the taxi column
    state[out[2] + 5] = 1
    # 4 bits for the passenger
    state[out[1] + 8] = 1
    # 4 bits for the destination
    state[out[0] + 12] = 1
    return state


if __name__ == "__main__":
    q_table = train_table() # Trains a Q-learning agent
    create_agent_weights(q_table) # Set neural network agent's weights according to Q-table
    #train_agent_weights(q_table) # Use this line to train the weights instead of infering them
    test_agent() # Test the agent with the neural network
