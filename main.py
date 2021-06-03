import gym
import numpy as np

env = gym.make("MountainCar-v0")

# Q-Learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 1000

DISCRETE_OS_SIZE = [10] * len(env.observation_space.high)
DISCRETE_OS_WIN_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state) -> object:
    DISCRETE_STATE = (state - env.observation_space.low) / DISCRETE_OS_WIN_SIZE
    return tuple(DISCRETE_STATE.astype(int))


for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    DISCRETE_STATE = get_discrete_state(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[DISCRETE_STATE])
        new_state, reward, done, _ = env.step(action)
        NEW_DISCRETE_STATE = get_discrete_state(new_state)
        if render:
            env.render()

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[NEW_DISCRETE_STATE])

            # Current Q value (for current state and performed action)
            current_q = q_table[DISCRETE_STATE + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT + max_future_q)

            # Update Q table with new Q value
            q_table[DISCRETE_STATE + (action,)] = new_q

        # Simulation ended (for any reason) - if goal position is achieved - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            print(f"Car reached the flag on episode {episode}")
            q_table[DISCRETE_STATE + (action,)] = 0

        DISCRETE_STATE = NEW_DISCRETE_STATE

env.close()
