import pygame

# Assuming constants.py has already been modified to include SCALE_FACTOR and scaled dimensions
from constants import ENV_HEIGHT, ENV_WIDTH, SCALE_FACTOR

# Apply scaling to the wall positions and dimensions
walls_mapping = [
    {
        "start_pos": (int(200 * SCALE_FACTOR), 100*SCALE_FACTOR),
        "width": int(60 * SCALE_FACTOR),
        "height": int(70 * SCALE_FACTOR),
    },
    # {"start_pos": (int(200 * SCALE_FACTOR), int(200 * SCALE_FACTOR)), "width": int(250 * SCALE_FACTOR), "height": int(10 * SCALE_FACTOR)},
    {
        "start_pos": (int(360 * SCALE_FACTOR), int(330 * SCALE_FACTOR)),
        "width": int(100 * SCALE_FACTOR),
        "height": int(60 * SCALE_FACTOR),
    },
    {
        "start_pos": (int(400 * SCALE_FACTOR), int(100 * SCALE_FACTOR)),
        "width": int(100 * SCALE_FACTOR),
        "height": int(60 * SCALE_FACTOR),
    },
    {
        "start_pos": (int(0 * SCALE_FACTOR), int(270 * SCALE_FACTOR)),
        "width": int(100 * SCALE_FACTOR),
        "height": int(100 * SCALE_FACTOR),
    },
    # {"start_pos": (int(0 * SCALE_FACTOR), int(400 * SCALE_FACTOR)), "width": int(100 * SCALE_FACTOR), "height": int(10 * SCALE_FACTOR)},
    # {
    #     "start_pos": (int(650 * SCALE_FACTOR), int(390 * SCALE_FACTOR)),
    #     "width": int(10 * SCALE_FACTOR),
    #     "height": int(220 * SCALE_FACTOR),
    # },
    # {"start_pos": (int(730 * SCALE_FACTOR), 0), "width": int(10 * SCALE_FACTOR), "height": int(250 * SCALE_FACTOR)},
    # {"start_pos": (0, int(420 * SCALE_FACTOR)), "width": int(210 * SCALE_FACTOR), "height": int(10 * SCALE_FACTOR)},
    # {"start_pos": (int(400 * SCALE_FACTOR), int(610 * SCALE_FACTOR)), "width": int(260 * SCALE_FACTOR), "height": int(10 * SCALE_FACTOR)},
    # {"start_pos": (int(650 * SCALE_FACTOR), int(390 * SCALE_FACTOR)), "width": int(230 * SCALE_FACTOR), "height": int(10 * SCALE_FACTOR)},
]


class Wall:
    def __init__(self, start_pos, width=10, height=10, color=(0, 0, 0)):
        self.start_pos = start_pos
        self.width = width
        self.height = height
        self.color = color
        self.rect = pygame.Rect(start_pos[0], start_pos[1], width, height)

    def draw(self, screen):
        """Draw the wall on the screen."""
        pygame.draw.rect(screen, self.color, self.rect)


# Example usage:
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))
    wall_objs = [Wall(**wall_data) for wall_data in walls_mapping]
    screen.fill((255, 255, 255))  # Clear screen with white at the start of each frame

    for wall in wall_objs:
        wall.draw(screen)

    running = True
    while running:
        pygame.display.flip()  # Update the full display Surface to the screen
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
