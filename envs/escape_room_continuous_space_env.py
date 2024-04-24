import gym
import numpy as np
import pygame
from gym import spaces

from constants import (
    CHECKPOINT_RADIUS,
    ENV_HEIGHT,
    ENV_WIDTH,
    MAX_WHEEL_VELOCITY,
    SCALE_FACTOR,
)
from robots.checkpoint import Checkpoint
from robots.robot import Robot
from robots.walls import Wall, walls_mapping
from utils.drawing_utils import draw_robot


class EscapeRoomEnv(gym.Env):
    def __init__(self, max_steps_per_episode=2000, goal= (530,290), delta= 15):
        super().__init__()

        self.spawn_x = int(70 * SCALE_FACTOR)
        self.spawn_y = int(70 * SCALE_FACTOR)
        
        # self.goal_position = np.array([int(70 * SCALE_FACTOR), int(450 * SCALE_FACTOR)])
        self.goal_position = np.array(
            [int(goal[0] * SCALE_FACTOR), int(goal[1] * SCALE_FACTOR)]
        )

        self.walls = [Wall(**wall_data) for wall_data in walls_mapping]
        # self.walls = []
        self.delta = delta
        self.goal = Checkpoint(self.goal_position, CHECKPOINT_RADIUS, (0, 128, 0), "G")

        low = np.array([-1.5 * ENV_WIDTH, -1.5 * ENV_HEIGHT, -np.pi, -5.0, -5.0, -5.0])
        high = np.array([1.5 * ENV_WIDTH, 1.5 * ENV_HEIGHT, np.pi, 5.0, 5.0, 5.0])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        self.robot = Robot((self.spawn_x, self.spawn_y))
        self.max_steps_per_episode = max_steps_per_episode
        self.t = 0  # Time step counter

        self.screen = None
        self.clock = None

    def step(self, action):
        action = np.clip(action, -1, +1).astype(np.float32)

        left_vel = action[0] * MAX_WHEEL_VELOCITY
        right_vel = action[1] * MAX_WHEEL_VELOCITY

        penalty, out_of_bounds = self.robot.update_and_check_collisions(
            left_vel, right_vel, self.walls, dt=1
        )
        reward = 0

        # Update the old distance
        new_pos = np.array([self.robot.x, self.robot.y])
        new_distance = np.linalg.norm(new_pos - np.array(self.goal.center_pos))
        # update the old distance

        alpha = 0.1
        distance_improvement = float(self.old_distance - new_distance)
        # print("Distance improvement improvement", distance_improvement)
        self.old_distance = new_distance

        # Calculate robot's heading and heading towards the goal
        goal_direction = np.array(self.goal.center_pos) - new_pos
        goal_angle = np.arctan2(goal_direction[1], goal_direction[0])
        heading_difference = self.robot.theta - goal_angle

        # Normalize the heading difference to the range [0, pi]
        heading_difference = (heading_difference + np.pi) % (2 * np.pi) - np.pi
        
        # Applying rewards based on heading difference
        if heading_difference > np.pi / 6:  # More than 30 degrees off
            reward += -np.log1p(heading_difference) 
            if self.robot.omega > np.pi/6:
                reward += -alpha
            
        if distance_improvement > 0:
            reward += +np.log1p(distance_improvement)
            if np.abs(left_vel) + np.abs(right_vel) < MAX_WHEEL_VELOCITY:
                reward += +alpha  # Adjust this value based on desired efficiency
        else:
            reward += -np.log1p(-distance_improvement)  # Adjust this value based on desired efficiency

        reward += penalty
        reward += -alpha  # steps penalty

        state = np.array(
            [
                self.robot.x,
                self.robot.y,
                self.robot.theta,
                self.robot.vx,
                self.robot.vy,
                self.robot.omega,
            ]
        )

        self.t += 1
        terminated = False
        truncated = False
        info = {}

        if self.goal.check_goal_reached((self.robot.x, self.robot.y), delta=self.delta):
            base_reward = +10_000
            efficiency_bonus = (np.log1p(self.max_steps_per_episode/self.t)) * base_reward * alpha  # to motivate agnet to reach the goal in fewer steps
            reward += base_reward + efficiency_bonus
            print(
                f"Goal '{self.goal.label}' reached in {self.t} steps with cumulative reward {reward} for this episode."
            )
            self.old_distance = np.linalg.norm(np.array([self.robot.x, self.robot.y])- np.array(self.goal.center_pos))
            terminated = True
            info["reason"] = "Goal_reached"
        elif out_of_bounds:
            terminated = True
            reward += -50
            info["reason"] = "out_of_bounds"
            # print(f"Robot went out of bounds after {self.t} steps with a cumulative reward of {reward}.")
        elif self.t >= self.max_steps_per_episode:
            truncated = True
            reward += -5
            info["reason"] = "max_steps_reached"
            # print(f"Max steps reached for this episode after {self.t} steps with a cumulative reward of {reward}.")

        return state, reward, terminated, truncated, info

    def reset(self):
        self.robot = Robot([self.spawn_x, self.spawn_y], init_angle=0)
        self.t = 0
        self.old_distance = np.linalg.norm(
            np.array([self.robot.x, self.robot.y])
            - np.array(self.goal.center_pos)
        )
        self.screen = None
        self.clock = None
        info = {"message": "Environment reset."}
        self.goal.reset()
        return (
            np.array(
                [
                    self.robot.x,
                    self.robot.y,
                    self.robot.theta,
                    self.robot.vx,
                    self.robot.vy,
                    self.robot.omega,
                ]
            ),
            info,
        )

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Clear the screen with white background at the start of each render cycle
        self.screen.fill((255, 255, 255))

        # Draw all walls
        for wall in self.walls:
            wall.draw(self.screen)

        # Draw the final goal
        self.goal.draw(self.screen)

        # Draw the robot on the screen
        draw_robot(self.screen, self.robot)

        if mode == "human":
            # Update the full display Surface to the screen for human viewing
            pygame.display.flip()
            # Limit the frame rate to maintain a consistent rendering speed
            self.clock.tick(30)
        elif mode == "rgb_array":
            # Capture the current rendered frame as an RGB array
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(frame, (1, 0, 2))  # Convert from (width, height, depth) to (height, width, depth)
            return frame

    def close(self):
        pygame.quit()


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
