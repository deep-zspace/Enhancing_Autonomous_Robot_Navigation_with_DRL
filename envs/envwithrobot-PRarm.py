import gym
from gym import spaces
import pygame
import numpy as np
import math

# Constants for environment and robot dimensions
ENV_WIDTH, ENV_HEIGHT = 1000, 800
ROBOT_RADIUS = 30
WHEEL_WIDTH = 10
WHEEL_HEIGHT = 20
WHEEL_BASE = ROBOT_RADIUS * 2 + WHEEL_WIDTH  # Distance between the two wheels
LINK_LENGTH_MIN, LINK_LENGTH_MAX = 50, 120  # Extensible link range


class DifferentialDriveRobot:
    def __init__(self, init_x, init_y, init_theta):
        self.x = init_x
        self.y = init_y
        self.theta = init_theta  # Orientation in radians

    def update_position(self, v1, v2, dt):
        v = (v1 + v2) / 2
        omega = (v1 - v2) / WHEEL_BASE
        self.theta += omega * dt
        self.theta %= 2 * math.pi
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt


class RobotArm:
    def __init__(self, base_theta, init_theta, init_q):
        self.base_theta = base_theta  # Base orientation relative to the robot's body
        self.theta = init_theta  # Arm's own orientation relative to its base
        self.q = init_q
        self.gripper_openness = 0.5

    def rotate_base(self, d_theta):
        self.base_theta = (self.base_theta + d_theta) % (2 * math.pi)

    def rotate_joint(self, alpha):
        self.theta = (self.theta + alpha) % (2 * math.pi)

    def extend_joint(self, dq):
        self.q = np.clip(self.q + dq, LINK_LENGTH_MIN, LINK_LENGTH_MAX)

    def adjust_gripper(self, openness):
        self.gripper_openness = np.clip(openness, 0, 1)


class EscapeRoomEnv(gym.Env):
    def __init__(self):
        super(EscapeRoomEnv, self).__init__()
        self.robot = DifferentialDriveRobot(400, 300, math.pi / 2)
        self.robot_arm = RobotArm(0, 0, 50)  # Initially aligned with the robot's body
        pygame.init()
        self.screen = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))
        self.clock = pygame.time.Clock()
        # Action space: wheel velocities, arm base rotation, arm joint rotation, extension, gripper openness
        self.action_space = spaces.Box(
            low=np.array([-10, -10, -math.pi, -math.pi, -10, 0], dtype=np.float32),
            high=np.array([10, 10, math.pi, math.pi, 10, 1], dtype=np.float32),
        )

    def step(self, action):
        v1, v2, d_base_theta, alpha, dq, gripper_openness = action
        self.robot.update_position(v1, v2, 1)
        self.robot_arm.rotate_base(d_base_theta)
        self.robot_arm.rotate_joint(alpha)
        self.robot_arm.extend_joint(dq)
        self.robot_arm.adjust_gripper(gripper_openness)
        return np.array([self.robot.x, self.robot.y, self.robot.theta]), 0, False, {}

    def reset(self):
        self.robot = DifferentialDriveRobot(400, 300, math.pi / 2)
        self.robot_arm = RobotArm(0, 0, 50)
        return np.array([self.robot.x, self.robot.y, self.robot.theta])

    def render(self, mode="human"):
        if mode == "human":
            self.screen.fill((255, 255, 255))
            self.draw_robot()
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        pygame.quit()

    def draw_robot(self):
        robot_center = (int(self.robot.x), int(self.robot.y))
        pygame.draw.circle(self.screen, (128, 128, 128), robot_center, ROBOT_RADIUS)
        # Calculate the angle for the wheels and the arm
        robot_theta_rad = math.radians(self.robot.theta)
        arm_theta_rad = (
            robot_theta_rad + self.robot_arm.base_theta + self.robot_arm.theta
        )
        # Draw wheels
        self.draw_wheels(robot_center, robot_theta_rad)
        # Draw arm
        arm_end = (
            robot_center[0] + int(self.robot_arm.q * math.cos(arm_theta_rad)),
            robot_center[1] + int(self.robot_arm.q * math.sin(arm_theta_rad)),
        )
        pygame.draw.line(self.screen, (0, 0, 255), robot_center, arm_end, 5)
        gripper_color = (
            255 * self.robot_arm.gripper_openness,
            0,
            255 * (1 - self.robot_arm.gripper_openness),
        )
        pygame.draw.circle(self.screen, gripper_color, arm_end, 10)

    def draw_wheels(self, robot_center, angle_rad):
        left_wheel_center = (
            robot_center[0] - (WHEEL_BASE / 2) * math.cos(angle_rad),
            robot_center[1] + (WHEEL_BASE / 2) * math.sin(angle_rad),
        )
        right_wheel_center = (
            robot_center[0] + (WHEEL_BASE / 2) * math.cos(angle_rad),
            robot_center[1] - (WHEEL_BASE / 2) * math.sin(angle_rad),
        )
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            pygame.Rect(
                left_wheel_center[0] - WHEEL_WIDTH / 2,
                left_wheel_center[1] - WHEEL_HEIGHT / 2,
                WHEEL_WIDTH,
                WHEEL_HEIGHT,
            ),
        )
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            pygame.Rect(
                right_wheel_center[0] - WHEEL_WIDTH / 2,
                right_wheel_center[1] - WHEEL_HEIGHT / 2,
                WHEEL_WIDTH,
                WHEEL_HEIGHT,
            ),
        )


# Initialize and run the environment
env = EscapeRoomEnv()
env.reset()

try:
    for _ in range(500):
        action = env.action_space.sample()
        env.step(action)
        env.render()
except KeyboardInterrupt:
    print("Simulation stopped manually.")
finally:
    env.close()
