import gym
import sys
import keras
import numpy as np
import random
import matplotlib.pyplot as plt
from random import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque


class Agent():
    def __init__(self, obs_space, action_space):
        self.obs_space = obs_space
        self.action_space = action_space
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.batch_size = 64
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
        model = Sequential()
        model.add(Dense(24, input_shape=(self.obs_space,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=[])
        return model

    def update_action(self, action_model, target_model):
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


def train():
    env = gym.make('CartPole-v1')
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]
    agent = Agent(observation_space, action_space)

    episodes = 600  # Games played in training phase
    max_steps = 500
    epsilon = 1
    epsilon_decay = 0.99
    epsilon_min = 0.05
    scores = []  # A list of all game scores
    recent_scores = []  # List that hold most recent 100 game scores

    for episode in range(episodes):
        observation = env.reset()
        for iteration in range(max_steps):
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
            if done:
                scores.append(iteration + 1)  # Append final score
                # Calculate recent scores
                if len(scores) > 100:
                    recent_scores = scores[-100:]
                # Print end-of-game information
                print(
                    "\rEpisode {:03d} , epsilon = {:.4f}, score = {:03d}".format(
                        episode, epsilon, iteration), end="")
                sys.stdout.flush()
                # Give reward on episode end to accelerate learning
                if iteration != 499:
                    reward = -5  # Give -5 reward for taking wrong action leading to failure
                if iteration == 499:
                    reward = 5  # Give +5 reward for completing the game successfully
                # Add the observation to replay memory
                agent.remember(old_observation, action, reward, None)
                break
            # Add the observation to replay memory
            agent.remember(old_observation, action, reward, observation)
            # Update the Deep Q-Network Model (only with a chance of 25% and
            # when the last score was worse than 495)
            if len(agent.memory) >= agent.batch_size and np.random.random(
            ) < 0.25 and scores[-1] < 495:
                agent.update_action(agent.model, agent.model)

        # If mean over the last 100 Games is >495, then success
        if np.mean(recent_scores) > 495 and iteration > 495:
            print("\nEnvironment solved in {} episodes.".format(episode), end="")
            break
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

    # Saving the model
    agent.model.save('models/cartpole_model.h5')

    # Plotting
    plt.plot(scores)
    plt.title('Training Phase')
    plt.ylabel('Time Steps')
    plt.ylim(ymax=510)
    plt.xlabel('Trial')
    plt.savefig('results/CartPoleTraining.png', bbox_inches='tight')
    plt.show()


def test():
    env = gym.make("CartPole-v1")
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]
    agent = Agent(observation_space, action_space)
    agent.load("models/cartpole_model.h5")
    scores = []

    # Playing 100 games
    for _ in range(100):
        obs = env.reset()
        episode_reward = 0
        while True:
            q_values = agent.get_q(obs)
            action = np.argmax(q_values)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        scores.append(episode_reward)

    # Plot the Performance
    plt.plot(scores)
    plt.title('Testing Phase')
    plt.ylabel('Time Steps')
    plt.ylim(ymax=510)
    plt.xlabel('Trial')
    plt.savefig('results/CartPoleTesting.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    train()  # Train new weights
    test()  # Use existing weights to play
