import gymnasium as gym
import numpy as np

def train_agent():
    # Initialize environment (is_slippery=False makes it deterministic for easier learning)
    env = gym.make("FrozenLake-v1", is_slippery=False)
    
    # Hyperparameters (TRY TWEAKING THEM!!!)
    num_episodes = 5000
    alpha = 0.5       # Learning rate
    gamma = 0.95      # Discount factor
    epsilon = 1.0     # Exploration rate
    min_epsilon = 0.01
    epsilon_decay = 0.995

    # Initialize Q-Table with zeros (16 states, 4 actions)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            # ==========================================
            # TODO 1: Epsilon-Greedy Action Selection
            # ==========================================
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()   # Explore: random action
            else:
                action = np.argmax(q_table[state])   # Exploit: best known action
            
            # Take the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # ==========================================
            # TODO 2: The Bellman Equation Update
            # Q(s,a) <- Q(s,a) + alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]
            # ==========================================
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            state = next_state
            
        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return q_table

if __name__ == "__main__":
    trained_q_table = train_agent()
    print("\nTraining complete! Final Q-Table:")
    print(trained_q_table)