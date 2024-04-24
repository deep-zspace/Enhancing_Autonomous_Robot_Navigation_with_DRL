import os
import gym
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange
from ppo_torch.ppo_agent import PPOAgent
from envs.escape_room_continuous_space_env import EscapeRoomEnv


def train_ppo_agent(n_episodes=5000, update_interval=500):
    env = EscapeRoomEnv(max_steps_per_episode=3000, goal=(300, 450))
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        K_epochs=40,
        eps_clip=0.2,
        total_updates=n_episodes // update_interval,
        action_std_init=0.6
    )

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"PPO_EscapeRoom_{n_episodes}_episodes"
    figure_file = f"{plot_dir}/{filename}.png"

    score_history = []
    loss_history = []

    pbar = trange(n_episodes, desc='Initializing training...')

    for i in pbar:
        state, info = env.reset()
        done = False
        score = 0
        steps = 0
        local_losses = []

        while not done:
            action, action_logprob = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.buffer.store(state, action, action_logprob, _, reward, done)

            score += reward
            state = next_state
            steps += 1

            if steps % update_interval == 0 or done:
                loss = agent.update()
                local_losses.append(loss)

        score_history.append(score)
        avg_loss = np.mean(local_losses) if local_losses else 0
        loss_history.append(avg_loss)

        description = f"Episode {i+1}: Score {score:.1f}, Avg Score {np.mean(score_history[-100:]):.3f}, Avg Loss {avg_loss:.4f}"
        pbar.set_description(description)

        if (i + 1) % (n_episodes // 10) == 0 or i == n_episodes - 1:
            agent.save(os.path.join(agent.checkpoint_dir, f'checkpoint_{i + 1}.pth'))
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=score_history)
            plt.title('Score per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.savefig(figure_file)
            plt.close()

    return {
        "no_of_episodes": n_episodes,
        "score_history": score_history,
        "loss_history": loss_history,
        "figure_file": figure_file
    }


if __name__ == '__main__':
    results = train_ppo_agent(n_episodes=5000, update_interval=500)
    print(results)