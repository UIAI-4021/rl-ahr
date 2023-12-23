import gym
import gym_maze
import numpy as np
import cv2

alpha = 0.1
gamma = 0.95
epsilon = 0.9
q_table = np.zeros(shape=(10, 10, 4))
policy = np.zeros(shape=(10, 10))


def getReward(state):
    if state[0] == 9 and state[1] == 9:
        return 5000
    return -1


def getAction(actions):
    if np.random.random() < epsilon:
        return np.argmax(actions)
    return np.random.randint(4)


def calculatePolicy(env):
    state = np.array([0, 0])
    env.reset()

    for __ in range(100000):
        action = getAction(q_table[state[0]][state[1]])
        old_state = state.copy()
        state, _, done, _ = env.step(action)
        reward = getReward(state, old_state)
        old_q_value = q_table[old_state[0]][old_state[1]][action]
        TD = reward + (gamma * np.max(q_table[state[0]][state[1]]) - old_q_value)
        q_table[old_state[0]][old_state[1]][action] += alpha * TD
        if done:
            env.reset()
            state = [0, 0]

    for i in range(10):
        for j in range(10):
            policy[i][j] = np.argmax(q_table[i][j])


if __name__ == "__main__":
    # Create an environment
    env = gym.make("maze-random-10x10-plus-v0")
    observation = env.reset()

    calculatePolicy(env)

    # Define the maximum number of iterations
    NUM_EPISODES = 1000

    for episode in range(NUM_EPISODES):

        # TODO: Implement the agent policy here
        # Note: .sample() is used to sample random action from the environment's action space

        # Choose an action (Replace this random action with your agent's policy)
        action = policy[int(observation[0])][int(observation[1])]

        # Perform the action and receive feedback from the environment
        next_state, reward, done, truncated = env.step(action)
        observation = next_state

        if done or truncated:
            observation = env.reset()
            print("GET GOAL!!!")
        # cv2.waitKey(200)
        env.render()

    # Close the environment
    env.close()
