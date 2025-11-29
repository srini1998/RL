import gymnasium as gym
import pygame
import numpy as np

WIDTH, HEIGHT = 800, 600

def world_to_screen(pos, cam_y):
    """Convert Box2D world coords to pygame screen coords."""
    x, y = pos
    screen_x = int(x * 30)
    screen_y = int((y - cam_y) * -30 + HEIGHT/2)
    return (screen_x, screen_y)

env = gym.make("LunarLander-v2", render_mode="rgb_array")
obs, info = env.reset()

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LunarLander Camera Fix")
clock = pygame.time.Clock()

done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Grab current frame + state
    frame = env.render()
    lander_y = obs[1]         # Y-position of lander from observation
    cam_y = lander_y - 5      # camera tracks the lander

    # Convert the numpy frame to a pygame surface
    surf = pygame.surfarray.make_surface(np.rot90(frame))

    # Draw to screen
    screen.blit(surf, (0, 0))
    pygame.display.update()

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    clock.tick(60)

env.close()
pygame.quit()

