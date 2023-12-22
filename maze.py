import gym
import gym_maze
import time
import numpy as np

X, Y = 10, 10
START = [0, 0]
GOAL = [9, 9]

class RLQ:
    def __init__(self, discount_factor, learning_rate):
        self.Q = np.zeros((X, Y, 4))
        self.pi = np.zeros((X, Y), dtype=int)
        self.gama = discount_factor
        self.learning_rate = learning_rate

        self.action_reward = -0.01
        self.goal_reward = 10

    def update_pi(self, state):
        # action is index_max
        action = max(range(len(self.Q[state[0], state[1]])), key=self.Q[state[0], state[1]].__getitem__)

        self.pi[state[0], state[1]] = action

    def policy(self, state):
        return self.pi[state[0], state[1]]

    def update(self, state, next_state, action):
        if (next_state[0] == GOAL[0] and next_state[1] == GOAL[1]):
            reward = self.goal_reward
        else:
            reward = self.action_reward
        
        state_q = self.Q[state[0], state[1], action]
        self.Q[state[0], state[1], action] = (1 - self.learning_rate) * state_q + self.learning_rate * (reward + self.gama*max(self.Q[next_state[0], next_state[1]]))

        self.update_pi(state)
    
    def print_pi(self):
        direction = {0: '^', 1: '>', 2: 'V', 3: '<'}
        for i in range(X):
            for j in range(Y):
                print(direction[self.
                      pi[i, j]], end=" ")
            print()
        print()

# Create an environment
env = gym.make("maze-random-10x10-plus-v0")
observation = env.reset()

state = START
agent = RLQ(discount_factor=0.99, learning_rate=0.3)

# Define the maximum number of iterations
NUM_EPISODES = 4000

for episode in range(NUM_EPISODES):
    env.render()

    # TODO: Implement the agent policy here
    # Note: .sample() is used to sample random action from the environment's action space

    # Choose an action (Replace this random action with your agent's policy)
    action = agent.policy(state)
    ac = {
        0: '^',
        1: '>',
        2: 'v',
        3: '<'
    }
    #print(ac[action])
    #time.sleep(0.5)
    # Perform the action and receive feedback from the environment
    next_state, reward, done, truncated = env.step(action)
    next_state = list(reversed(next_state))
    agent.update(state=state, next_state=next_state, action=action)
    state = next_state

    if done or truncated:
        observation = env.reset()
        state = START
    

agent.print_pi()
time.sleep(1000)

# Close the environment
env.close()