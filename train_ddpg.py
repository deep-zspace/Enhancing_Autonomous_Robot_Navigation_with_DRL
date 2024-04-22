import os
import gym
from matplotlib import pyplot as plt
import numpy as np
from ddpg_torch.ddpg_agent import Agent
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


def train_diff_robot_custom_env(alpha=0.0001, beta=0.001, tau=0.001, n_games=100):
    env = EscapeRoomEnv()
    agent = Agent(
        alpha=alpha,
        beta=beta,
        input_dims=env.observation_space.shape,
        tau=tau,
        batch_size=64,
        fc1_dims=400,
        fc2_dims=300,
        n_actions=env.action_space.shape[0],
    )

    filename = f"EscapeRoom_alpha_{agent.alpha}_beta_{agent.beta}_{n_games}_games"
    figure_file = f"plots/{filename}.png"
    score_history = []
    critic_losses = []
    actor_losses = []

    save_interval = n_games // 10  # Save model every 10% of n_games
    pbar = trange(n_games)

    for i in pbar:
        state, info = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            critic_loss, actor_loss = agent.learn()
            score += reward
            state = next_state

        score_history.append(score)
        critic_losses.append(critic_loss if critic_loss else 0)
        actor_losses.append(actor_loss if actor_loss else 0)

        avg_score = np.mean(
            score_history
        )  # Calculate average score after appending current score

        if i % save_interval == 0 or i == n_games - 1:  # Save model at intervals
            agent.save_models()
            print(f"Model saved at episode {i}")

        pbar.set_description(
            f"Episode {i}: Score {score:.1f}, Average Score {avg_score:.1f}"
        )

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, critic_losses, actor_losses, figure_file)


if __name__ == "__main__":
    train_diff_robot_custom_env()
