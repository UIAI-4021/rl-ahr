import gym
import gym_maze
import numpy as np
import cv2

learning_rate = 0.1
discount_factor = 0.95
epsilon = 0.9
# First Policy Attribution
q_table = np.zeros(shape=(10, 10, 4))
policy = np.zeros(shape=(10, 10))
# Second Policy attribution
second_policy = np.zeros(shape=(10, 10))
environment_rows = 10
environment_columns = 10
q_values = np.zeros((environment_rows, environment_columns, 4))

# Second Policy next action
def get_next_action(current_row_index, current_column_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:
        return np.random.randint(4)

# Policy calculation for second Policy
def calculatePolicy_second(env):
    NUM_EPISODES = 1000
    for episode in range(NUM_EPISODES):
        observation = env.reset()
        row_index = observation[0]
        column_index = observation[1]

        done = False
        while not done:
            action_index = get_next_action(int(row_index), int(column_index), epsilon)
            # TODO: Implement the agent policy here
            next_state, reward, done, truncated = env.step(action_index)
            reward = getReward(next_state)
            if truncated:
                print("truncated")
                print(f"truncated next state{next_state}")
                break
            row_index_old, column_index_old = row_index, column_index
            old_q_value = q_values[int(row_index_old), int(column_index_old), action_index]
            row_index = next_state[0]
            column_index = next_state[1]
            temporal_difference = reward + (
                        discount_factor * np.max(q_values[int(row_index), int(column_index)])) - old_q_value
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[int(row_index_old), int(column_index_old), action_index] = new_q_value

    for i in range(10):
        for j in range(10):
            second_policy[i][j] = np.argmax(q_values[i][j])

    env.reset()
    return


# Getting reward for both policies
def getReward(state):
    if state[0] == 9 and state[1] == 9:
        return 5000
    return -1

# First Policy next action
def getAction(actions):
    if np.random.random() < epsilon:
        return np.argmax(actions)
    return np.random.randint(4)


# Policy calculation for first Policy
def calculatePolicy(env):
    state = np.array([0, 0])
    env.reset()

    for __ in range(100000):
        action = getAction(q_table[state[0]][state[1]])
        old_state = state.copy()
        state, _, done, _ = env.step(action)
        reward = getReward(state)
        old_q_value = q_table[old_state[0]][old_state[1]][action]
        TD = reward + (discount_factor * np.max(q_table[state[0]][state[1]]) - old_q_value)
        q_table[old_state[0]][old_state[1]][action] += learning_rate * TD
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

    # Second policy ( Uncomment to see what happens)
    #calculatePolicy_second(env)

    # Define the maximum number of iterations
    NUM_EPISODES = 1000

    for episode in range(NUM_EPISODES):

        # TODO: Implement the agent policy here
        # Note: .sample() is used to sample random action from the environment's action space

        # action based on first policy
        action = policy[int(observation[0])][int(observation[1])]

        # action based on second Policy
        #action = second_policy[int(observation[0])][int(observation[1])]

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
