import os
import gym
import numpy as np
from matplotlib import pyplot as plt
from td3_torch import Agent  # Ensure this import matches your TD3 agent implementation
from envs.escape_room_continuous_space_env import EscapeRoomEnv
from tqdm import trange

def plot_learning_curve(x, scores, critic_losses, actor_losses, figure_file):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Scores plot
    ax1.plot(x, scores, label="Score per Episode")
    ax1.set_title("Learning Curve for Scores")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score")
    ax1.legend()
    ax1.grid(True)

    # Critic loss plot
    ax2.plot(x, critic_losses, color="red", label="Critic Loss")
    ax2.set_title("Learning Curve for Critic Loss")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    # Actor loss plot
    ax3.plot(x, actor_losses, color="green", label="Actor Loss")
    ax3.set_title("Learning Curve for Actor Loss")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Loss")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    plt.savefig(figure_file)
    plt.close()

def train_td3(alpha=0.001, beta=0.001, tau=0.005, n_games=1000):
    env = EscapeRoomEnv()
    agent = Agent(
        alpha=alpha,
        beta=beta,
        input_dims=env.observation_space.shape,
        tau=tau,
        batch_size=100,
        layer1_size=400,
        layer2_size=300,
        n_actions=env.action_space.shape[0]
    )

    filename = f"TD3_EscapeRoom_{n_games}_games"
    figure_file = f"plots/{filename}.png"
    score_history = []
    critic_losses = []
    actor_losses = []

    pbar = trange(n_games)
    for i in pbar:
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            critic_loss, actor_loss = agent.learn()
            score += reward
            state = next_state

        score_history.append(score)
        critic_losses.append(critic_loss)
        actor_losses.append(actor_loss)

        avg_score = np.mean(score_history[-100:])
        pbar.set_description(f"Episode {i}: Score {score:.2f}, Avg Score {avg_score:.2f}")

        if (i + 1) % (n_games // 10) == 0 or i == n_games - 1:
            agent.save_models()
            print(f"Model saved at episode {i}")

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, critic_losses, actor_losses, figure_file)

if __name__ == "__main__":
    train_td3()
