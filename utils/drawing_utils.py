import math

import pygame

from constants import ROBOT_RADIUS, WHEEL_HEIGHT, WHEEL_WIDTH
from robots.robot import Robot


def draw_robot(screen, robot):
    pygame.draw.circle(screen, (128, 128, 128), (robot.x, robot.y), ROBOT_RADIUS)
    draw_wheels(screen, robot)
    # draw_link(screen, robot)


def draw_wheels(screen, robot):
    rad_angle = robot.theta  # Robot's orientation in radians

    # Calculate the offsets for the wheels correctly
    # Assuming the wheels are perpendicular to the robot's orientation
    wheel_offset_x = ROBOT_RADIUS + WHEEL_WIDTH / 2
    wheel_offset_y = WHEEL_HEIGHT / 2

    left_wheel_center = [
        robot.x
        - wheel_offset_x * math.sin(rad_angle),  # Corrected for perpendicular offset
        robot.y + wheel_offset_x * math.cos(rad_angle),
    ]
    right_wheel_center = [
        robot.x
        + wheel_offset_x * math.sin(rad_angle),  # Corrected for perpendicular offset
        robot.y - wheel_offset_x * math.cos(rad_angle),
    ]

    # Draw left and right wheels
    draw_wheel(
        screen, left_wheel_center, rad_angle + math.pi / 2
    )  # Rotate wheel surface by 90 degrees
    draw_wheel(
        screen, right_wheel_center, rad_angle + math.pi / 2
    )  # Rotate wheel surface by 90 degrees


def draw_wheel(screen, center, angle):
    wheel_surf = pygame.Surface((WHEEL_WIDTH, WHEEL_HEIGHT), pygame.SRCALPHA)
    pygame.draw.rect(wheel_surf, (0, 0, 0), [0, 0, WHEEL_WIDTH, WHEEL_HEIGHT])
    rotated_surf = pygame.transform.rotate(wheel_surf, -math.degrees(angle))
    screen.blit(rotated_surf, rotated_surf.get_rect(center=center))


def draw_link(screen, robot: Robot):
    rad_angle = math.radians(robot.theta + robot.servo_angle)
    link_end = [
        robot.x + robot.link_length * math.cos(rad_angle),
        robot.y + robot.link_length * math.sin(rad_angle),
    ]
    gripper_color = (255, 0, 0) if robot.gripper_closed else (0, 0, 255)
    pygame.draw.line(screen, (0, 255, 0), (robot.x, robot.y), link_end, 5)
    pygame.draw.circle(screen, gripper_color, link_end, 8)
