import numpy as np
import pygame


class Checkpoint:
    def __init__(self, center_pos, radius, color, label, font_size=36):
        self.center_pos = center_pos
        self.radius = radius
        self.original_color = color
        self.color = color
        self.label = label
        self.font_size = font_size
        self.reached = False

    def draw(self, screen):
        # Use circle for checkpoint visualization
        pygame.draw.circle(screen, self.color, self.center_pos, self.radius)
        if not self.reached:  # Only draw label if not reached
            font = pygame.font.Font(None, self.font_size)
            text = font.render(self.label, True, (255, 255, 255))
            text_rect = text.get_rect(center=self.center_pos)
            screen.blit(text, text_rect)

    def check_goal_reached(self, robot_pos, delta=10):
        """Check if the robot has reached this checkpoint.
        Args:
            robot_pos (tuple): The (x, y) position of the robot.
            delta (float): The acceptable distance to the checkpoint to consider it reached.

        Returns:
            bool: True if the checkpoint is reached, otherwise False.
        """
        if isinstance(robot_pos, np.ndarray):
            robot_pos = robot_pos.tolist()
        distance = np.linalg.norm(np.array(self.center_pos) - np.array(robot_pos))
        # distance = pygame.math.Vector2(robot_pos).distance_to(pygame.math.Vector2(self.center_pos))
        # distance = np.sqrt(
        #     (robot_pos[0] - self.center_pos[0]) ** 2
        #     + (robot_pos[1] - self.center_pos[1]) ** 2
        # )
        # print(f"Distance to checkpoint {self.label} :: {distance}")
        if distance <= delta:
            if not self.reached:
                self.reached = (
                    True  # Mark as reached and update the color to indicate this
                )
                self.color = (0, 255, 0)  # Change color to green when reached
            return True
        return False

    def reset(self):
        """Reset the color of the checkpoint to its original color."""
        self.color = self.original_color
        self.reached = False


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))

    # Define checkpoints
    checkpoints = [
        Checkpoint(center_pos=(200, 150), radius=30, color=(255, 0, 0), label="A"),
        Checkpoint(center_pos=(400, 300), radius=30, color=(0, 0, 255), label="B"),
        Checkpoint(center_pos=(600, 450), radius=30, color=(0, 255, 0), label="C"),
    ]

    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))  # Clear screen
        robot_position = (
            pygame.mouse.get_pos()
        )  # For demonstration, use mouse position as robot position

        # Draw and check each checkpoint
        for checkpoint in checkpoints:
            checkpoint.draw(screen)
            if checkpoint.check_goal_reached(robot_position):
                print(f"Reached {checkpoint.label}")

        pygame.display.flip()  # Update display

    pygame.quit()
