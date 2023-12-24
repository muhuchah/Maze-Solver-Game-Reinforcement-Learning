import gym
import gym_maze
import time
import random
import numpy as np

X, Y = 10, 10
START = [0, 0]
GOAL = [9, 9]
EPSILON = 0.1

class RLQ:

    def __init__(self, discount_factor, learning_rate):
        self.Q = np.zeros((X, Y, 4))
        self.pi = np.zeros((X, Y), dtype=int)
        self.gama = discount_factor
        self.learning_rate = learning_rate

        self.action_reward = -0.001
        self.goal_reward = 1


    def update(self, state, next_state, action):
        if (next_state[0] == GOAL[0] and next_state[1] == GOAL[1]):
            reward = self.goal_reward
        else:
            reward = self.action_reward
        
        state_q = self.Q[state[0], state[1], action]
        self.Q[state[0], state[1], action] = (1 - self.learning_rate) * state_q + self.learning_rate * (reward + self.gama*max(self.Q[next_state[0], next_state[1]]))

        self.update_pi(state)


    def update_pi(self, state):
        # action is index_max
        action = max(range(len(self.Q[state[0], state[1]])), key=self.Q[state[0], state[1]].__getitem__)

        self.pi[state[0], state[1]] = action


    def policy(self, state):
        return self.pi[state[0], state[1]]
    

    def print_pi(self):
        direction = {0: '^', 1: 'v', 2: '>', 3: '<'}
        for i in range(X):
            for j in range(Y):
                print(direction[self.
                      pi[i, j]], end=" ")
            print()
        print()
    

    def print_q(self):
        for i in range(X):
            for j in range(Y):
                action = max(range(len(self.Q[i, j])), key=self.Q[i, j].__getitem__)
                ac = {0: '^', 1: 'v', 2: '>', 3: '<'}
                print(round(max(self.Q[i, j]), 2), end=f'{ac[action]}\t')
            print()
        print()


# Create an environment
env = gym.make("maze-random-10x10-plus-v0")
observation = env.reset()


state = START
agent = RLQ(discount_factor=0.95, learning_rate=0.1)

# Define the maximum number of iterations
NUM_EPISODES_TRAIN = 500000
NUM_EPISODES = 1000

for episode in range(NUM_EPISODES_TRAIN):
    rnd = random.random()
    if rnd < EPSILON:
        action = random.randint(0, 3)
    else:
        action = agent.policy(state)
    
    next_state, reward, done, truncated = env.step(action)

    next_state = list(reversed(next_state))
    agent.update(state=state, next_state=next_state, action=action)
    state = next_state

    if done or truncated:
        observation = env.reset()
        state = START
        EPSILON = 0.99 * EPSILON
        

wins = 0
for episode in range(NUM_EPISODES):
    env.render()

    action = agent.policy(state)

    next_state, reward, done, truncated = env.step(action)
    time.sleep(0.1)
    
    next_state = list(reversed(next_state))
    state = next_state

    if done:
        wins += 1
    if done or truncated:
        observation = env.reset()
        state = START
    

print(f"wins: {wins}")
agent.print_pi()
agent.print_q()
time.sleep(60)

# Close the environment
env.close()
