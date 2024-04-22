import math
import time

import gym
import numpy as np
import pygame
from gym import spaces

# Constants for environment and robot dimensions
ENV_WIDTH, ENV_HEIGHT = 1000, 800
ROBOT_RADIUS = 30
WHEEL_WIDTH = 10
WHEEL_HEIGHT = 20
WHEEL_OFFSET = (
    ROBOT_RADIUS + WHEEL_HEIGHT / 2
)  # Offset from the center of the robot to the wheel's center
LINK_LENGTH_MIN, LINK_LENGTH_MAX = 50, 120  # Min and max lengths of the link


class DifferentialDriveRobot:
    def _init_(self, init_x, init_y, init_theta):
        self.x = init_x
        self.y = init_y
        self.theta = init_theta  # Orientation in radians

    def update_position(self, v1, v2, dt=1):
        # Compute the robot velocity and angular velocity from the wheel velocities
        # Linear velocity: V = (VL + VR) / 2
        # Angular velocity: ω = (VR - VL) / W

        v = (v1 + v2) / 2
        omega = WHEEL_RADIUS * (v1 - v2) / WHEEL_BASE

        # Update the robot's orientation
        self.theta += omega * dt
        self.theta %= 2 * math.pi  # Normalize theta

        # Update the robot's position
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt


class RobotArm:
    def _init_(self, init_theta, init_q):
        self.theta = init_theta
        self.q = init_q
        self.gripper_open = True

    def rotate_joint(self, d_theta):
        self.theta += d_theta
        self.theta %= 2 * math.pi

    def extend_joint(self, dq):
        self.q += dq
        self.q = max(LINK_LENGTH_MIN, min(self.q, LINK_LENGTH_MAX))

    def toggle_gripper(self):
        self.gripper_open = not self.gripper_open


class EscapeRoomEnv(gym.Env):
    def _init_(self):
        super(EscapeRoomEnv, self)._init_()
        self.robot = DifferentialDriveRobot(400, 300, 0)
        self.robot_arm = RobotArm(0, 50)
        pygame.init()
        self.screen = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))
        self.clock = pygame.time.Clock()
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -0.1, -10, 0]),
            high=np.array([1, 1, 0.1, 10, 1]),
            dtype=np.float32,
        )

    def step(self, action):
        v1, v2, d_theta, dq, gripper_action = action
        self.robot.update_position(v1, v2, 0.1)  # dt = 0.1 second
        self.robot_arm.rotate_joint(d_theta)
        self.robot_arm.extend_joint(dq)
        if gripper_action > 0.5:
            self.robot_arm.toggle_gripper()
        return (
            np.array(
                [
                    self.robot.x,
                    self.robot.y,
                    self.robot.theta,
                    self.robot_arm.q,
                    self.robot_arm.gripper_open,
                ]
            ),
            0,
            False,
            {},
        )

    def reset(self):
        self.robot = DifferentialDriveRobot(400, 300, 0)
        self.robot_arm = RobotArm(0, 50)
        return np.array(
            [
                self.robot.x,
                self.robot.y,
                self.robot.theta,
                self.robot_arm.q,
                self.robot_arm.gripper_open,
            ]
        )

    def render(self, mode="human"):
        if mode == "human":
            self.screen.fill((255, 255, 255))
            self.draw_robot(self.screen)
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        pygame.quit()

    def draw_robot(self, screen):
        # Robot body center
        robot_center = (int(self.robot.x), int(self.robot.y))

        # Draw the robot body
        pygame.draw.circle(screen, (128, 128, 128), robot_center, ROBOT_RADIUS)

        # Wheels
        wheel_angle = math.radians(self.robot.theta)
        left_wheel_center = (
            robot_center[0] - WHEEL_OFFSET * math.sin(wheel_angle),
            robot_center[1] + WHEEL_OFFSET * math.cos(wheel_angle),
        )
        right_wheel_center = (
            robot_center[0] + WHEEL_OFFSET * math.sin(wheel_angle),
            robot_center[1] - WHEEL_OFFSET * math.cos(wheel_angle),
        )

        pygame.draw.rect(
            screen,
            (0, 0, 0),
            (
                left_wheel_center[0] - WHEEL_WIDTH // 2,
                left_wheel_center[1] - WHEEL_HEIGHT // 2,
                WHEEL_WIDTH,
                WHEEL_HEIGHT,
            ),
        )
        pygame.draw.rect(
            screen,
            (0, 0, 0),
            (
                right_wheel_center[0] - WHEEL_WIDTH // 2,
                right_wheel_center[1] - WHEEL_HEIGHT // 2,
                WHEEL_WIDTH,
                WHEEL_HEIGHT,
            ),
        )

        # Draw the arm and gripper
        arm_end_x = robot_center[0] + self.robot_arm.q * math.cos(
            math.radians(self.robot.theta + self.robot_arm.theta)
        )
        arm_end_y = robot_center[1] + self.robot_arm.q * math.sin(
            math.radians(self.robot.theta + self.robot_arm.theta)
        )
        arm_end = (int(arm_end_x), int(arm_end_y))
        pygame.draw.line(screen, (0, 0, 255), robot_center, arm_end, 5)

        gripper_color = (255, 0, 0) if self.robot_arm.gripper_open else (0, 255, 0)
        pygame.draw.circle(screen, gripper_color, arm_end, 10)


# Initialize and run the environment
env = EscapeRoomEnv()
env.reset()

try:
    for _ in range(500):
        action = env.action_space.sample()
        env.step(action)
        env.render()
        time.sleep(0.05)
except KeyboardInterrupt:
    print("Simulation stopped manually.")
finally:
    env.close()
