'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# 1. Initialize Environment
# Removed render_mode="human" to significantly speed up the training process
env = gym.make("MountainCar-v0") 

# 2. Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = 500

# Epsilon Parameters (Exploration vs Exploitation Strategy)
epsilon = 1.0  # Start with 100% exploration
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# 3. Discretization (Converting continuous state space to discrete grid)
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Initialize Q-Table with random values between -2 and 0
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    """Helper function to convert continuous state into a discrete tuple."""
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

# Data structures to store metrics for the final report chart
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

# 4. Training Loop
for episode in range(EPISODES):
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    episode_reward = 0
    
    while not done:
        # Action Selection: EPSILON-GREEDY STRATEGY
        if np.random.random() > epsilon:
            # Exploitation: take the best known action for the current state
            action = np.argmax(q_table[discrete_state])
        else:
            # Exploration: take a random action to discover new paths
            action = env.action_space.sample()

        # Execute the action in the environment
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        new_discrete_state = get_discrete_state(new_state)

        # 5. The Bellman Equation (Updating the Q-Table)
        if not done:
            # Maximum possible Q-value in the next step
            max_future_q = np.max(q_table[new_discrete_state])
            
            # Current Q-value
            current_q = q_table[discrete_state + (action,)]
            
            # Calculate the new Q-value using the Bellman equation
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            
            # Update the Q-table with the new value
            q_table[discrete_state + (action,)] = new_q
            
        elif new_state[0] >= env.unwrapped.goal_position:
            # If the goal is reached, the Q-value for this action becomes 0 (highest reward)
            q_table[discrete_state + (action,)] = 0
            
        # Move to the next state
        discrete_state = new_discrete_state

    # Epsilon Decay: Gradually reduce exploration over time
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # Record the total reward for the current episode
    ep_rewards.append(episode_reward)
    
    # Calculate and store aggregated metrics every 'SHOW_EVERY' episodes
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f"Episode: {episode:>5d}, Average Reward: {average_reward:>4.1f}, Current Epsilon: {epsilon:>1.2f}")

env.close()

# 6. Chart Generation (To be included in the final PDF report)
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="Average Reward")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="Max Reward")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="Min Reward")
plt.legend(loc=4)
plt.title("NeoDrive Agent Performance (Q-Learning)")
plt.xlabel("Training Episodes")
plt.ylabel("Reward Score")
plt.show()
