import pygame

from envs.escape_room_env import EscapeRoomEnv


def run_environment():
    pygame.init()
    env = EscapeRoomEnv()
    env.reset()

    running = True
    while running:
        env.render()  # Update the visual display
        pygame.event.pump()  # Update the internal buffer of events (important for detecting continuous presses)

        # Check for continuous key presses
        keys = pygame.key.get_pressed()
        action = 2  # Default action is stop (2)
        if keys[pygame.K_UP]:
            action = 0  # Forward
        elif keys[pygame.K_DOWN]:
            action = 1  # Backward
        elif keys[pygame.K_LEFT]:
            action = 3  # Rotate Left
        elif keys[pygame.K_RIGHT]:
            action = 4  # Rotate Right

        state, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print(f"Action: {action}, State: {state}, Reward: {reward}, Info: {info}")
            print("Episode terminated")
            env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()  # Update the full display Surface to the screen
        env.clock.tick(30)  # Limit the frame rate

    env.close()


if __name__ == "__main__":
    run_environment()
