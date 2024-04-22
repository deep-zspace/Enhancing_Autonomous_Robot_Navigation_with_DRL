import gym
from envs.escape_room_continuous_space_env import EscapeRoomEnv


if __name__ == "__main__":
    env = EscapeRoomEnv()
    assert (
        isinstance(env.observation_space, gym.spaces.Box)
        and len(env.observation_space.shape) == 1
    )
    try:
        for _ in range(1000):
            action = env.action_space.sample()
            env.step(action)
            env.render()
    except KeyboardInterrupt:
        print("Simulation stopped manually.")
    finally:
        env.close()
