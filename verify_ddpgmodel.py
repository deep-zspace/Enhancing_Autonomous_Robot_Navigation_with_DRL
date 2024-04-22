import gym
import numpy as np

from envs.escape_room_continuous_space_env import EscapeRoomEnv
from ddpg_torch.ddpg_agent import Agent


def load_and_simulate(env, agent, n_episodes=5, max_steps=500):
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            env.render()  # Render the environment to visualize the agent's behavior
            action = agent.choose_action(
                state
            )  # Agent selects an action based on the current state
            state, reward, terminated, truncated, info = env.step(
                action
            )  # Execute the action in the environment
            done = terminated or truncated
            total_reward += reward
            steps += 1

        print(f"Episode {episode + 1}: Total reward = {total_reward}, Steps = {steps}")
        if "next_goal" in info:
            print(f"Moving to Next Goal: {info['next_goal']}")

        if steps >= max_steps:
            print(f"Episode {episode + 1} reached the maximum of {max_steps} steps.")

    env.close()  # Close the environment when done


def main():
    env = EscapeRoomEnv(max_steps_per_episode=500)
    n_actions = env.action_space.shape[0]
    input_dims = env.observation_space.shape

    # Parameters used during training, for consistency
    alpha = 0.0001
    beta = 0.001
    tau = 0.001
    fc1_dims = 400
    fc2_dims = 300

    agent = Agent(
        alpha=alpha,
        beta=beta,
        input_dims=input_dims,
        tau=tau,
        fc1_dims=fc1_dims,
        fc2_dims=fc2_dims,
        n_actions=n_actions,
        batch_size=64,  # Batch size might not be necessary for simulation but required for initialization
    )

    # Load models from the appropriate file paths
    agent.load_models()  # Ensure this matches your checkpoint saving logic or specify the paths if needed

    load_and_simulate(env, agent, n_episodes=5, max_steps=1000)


if __name__ == "__main__":
    main()
