'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
import gymnasium as gym
import numpy as np

# 1. Initialize Environment
env = gym.make("MountainCar-v0", render_mode="human")

# 2. Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 1000
SHOW_EVERY = 200
STATS_EVERY = 100

# 3. Discretization (Turning continuous numbers into a grid)
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Initialize Q-Table with random values
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

# 4. Training Loop
for episode in range(EPISODES):
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    
    while not done:
        # Action Selection (Exploitation)
        action = np.argmax(q_table[discrete_state])
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        new_discrete_state = get_discrete_state(new_state)

        # 5. The Bellman Equation (The "Learning" part)
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0 # Goal reached
            
        discrete_state = new_discrete_state

    if episode % STATS_EVERY == 0:
        print(f"Episode: {episode} completed.")

env.close()