'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt # Ajout pour générer les graphiques exigés

# 1. Initialize Environment
env = gym.make("MountainCar-v0") # Enlevé render_mode="human" pour accélérer l'entraînement

# 2. Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000 # Augmenté un peu pour MountainCar
SHOW_EVERY = 500

# Paramètres Epsilon (Exploration vs Exploitation) - REQUIS PAR LE DEVOIR
epsilon = 1.0  # Exploration à 100% au début
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# 3. Discretization
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Initialize Q-Table with random values
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

# Pour stocker les métriques exigées par le rapport
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

# 4. Training Loop
for episode in range(EPISODES):
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    episode_reward = 0
    
    while not done:
        # Action Selection: EPSILON-GREEDY STRATEGY (Correction majeure)
        if np.random.random() > epsilon:
            # Exploitation: prendre la meilleure action connue
            action = np.argmax(q_table[discrete_state])
        else:
            # Exploration: prendre une action au hasard
            action = env.action_space.sample()

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        new_discrete_state = get_discrete_state(new_state)

        # 5. The Bellman Equation
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            
            # Équation de mise à jour Q-Learning
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
            
        elif new_state[0] >= env.unwrapped.goal_position: # Ajusté pour la nouvelle version de Gym
            q_table[discrete_state + (action,)] = 0 # Goal reached
            
        discrete_state = new_discrete_state

    # Epsilon Decay (Réduire l'exploration au fil du temps)
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # Enregistrement des métriques
    ep_rewards.append(episode_reward)
    
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f"Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}")

env.close()

# 6. Génération du graphique (À inclure dans votre rapport PDF)
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="Average Reward")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="Max Reward")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="Min Reward")
plt.legend(loc=4)
plt.title("Performances de l'Agent NeoDrive (Q-Learning)")
plt.xlabel("Épisodes")
plt.ylabel("Récompense")
plt.show()
