import gym
import gym_maze
import numpy as np
import cv2


alpha = 0.1
gamma = 0.95
epsilon = 0.9
q_table = np.zeros(shape=(10, 10, 4))
policy = np.zeros(shape=(10, 10))


def getAction(actions):
    if np.random.random() < epsilon:
        return np.argmax(actions)
    return np.random.randint(4)


if __name__ == "__main__":
    # Create an environment
    env = gym.make("maze-random-10x10-plus-v0")
    observation = env.reset()

    # Define the maximum number of iterations
    NUM_EPISODES = 1000

    for episode in range(NUM_EPISODES):

        # TODO: Implement the agent policy here
        # Note: .sample() is used to sample random action from the environment's action space

        # Choose an action (Replace this random action with your agent's policy)
        action = env.action_space.sample()

        # Perform the action and receive feedback from the environment
        next_state, reward, done, truncated = env.step(action)

        if done or truncated:
            observation = env.reset()
        cv2.waitKey(200)
        env.render()

    # Close the environment
    env.close()
