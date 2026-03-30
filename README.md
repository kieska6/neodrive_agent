NeoDrive-Q: Autonomous Driving Agent (Q-Learning Project)Project Overview

NeoDrive-Q is an autonomous driving agent developed as a demonstration of a core Reinforcement Learning (RL) project. The agent's primary purpose is to navigate the simulated 2D MountainCar-v0 environment provided by Gymnasium. By optimizing its driving strategy through trial and error, the agent aims to reach the designated goal while minimizing energy consumption and avoiding inefficient paths.Technical ImplementationAlgorithm1

The agent implements Q-Learning, which is a model-free, off-policy reinforcement learning algorithm.1
Mechanism: The algorithm uses a multi-dimensional Q-Table to store the expected future rewards for every state-action pair.1
Strategy: An Epsilon-Greedy strategy is employed to balance the exploration of new, random actions with the exploitation of the best-known actions in the Q-Table.1
Learning Logic: The Q-value is updated using the Bellman Equation.1
Environment Specifications (MountainCar-v0)

The environment and its state space were configured as follows:
| Characteristic   | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| Source           | Gymnasium (formerly OpenAI Gym)                                             |
| Features         | 2 primary observations: Car Position (horizontal coordinate) and Car Velocity (speed/direction). |
| Actions          | 3 discrete actions: 0 (Accelerate Left), 1 (No Acceleration), 2 (Accelerate Right). |
| Training Volume  | 1,000 training episodes                                                     |

 	
Preprocessing

Since the state space (position and velocity) is continuous, Discretization (Binning) was applied. This technique converts the continuous values into a discrete grid, which is necessary for efficient storage and updates within the Q-Table.Libraries and Toolkits1
Gymnasium: Provides the simulated environment and standard RL API.1
NumPy: Used for creating, indexing, and updating the multidimensional Q-Table.1
Matplotlib: (Optional) Used for plotting performance metrics.1
Pickle: Used to serialize and save the trained Q-Table.1
Results and InsightsPerformance
The agent successfully learns the necessary strategy: it realizes it must drive backward during initial episodes to gain enough momentum to climb the forward hill and reach the goal.1
The success rate improves significantly by episode 500.1
Upon completion of training, the agent consistently reaches the target flag in under 120 steps.1
Future Work

The current Q-Learning implementation faces the "Curse of Dimensionality". As the number of features increases (e.g., in a real-world scenario), the Q-Table size would grow exponentially. The logical next step for scaling this project would be to migrate to a Deep Q-Network (DQN), which uses a Neural Network to replace the Q-Table, allowing it to handle continuous and high-dimensional state spaces.Installation and ExecutionPrerequisites1

This project requires Python 3 and the following dependencies:
pip install gymnasium numpy matplotlib 
Running the Agent
Training: Execute the main script to start the Q-Learning process:
python neodrive_agent.py
The console will print the training progress every 100 episodes.1
Testing: To visualize the agent in action after training, modify the script to enable the environment rendering mode.1
Author

This work was completed by Kieran Kenga
