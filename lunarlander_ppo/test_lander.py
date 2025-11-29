import gymnasium as gym
from stable_baselines3 import PPO
import pygame
import numpy as np

WIDTH, HEIGHT = 800, 600

def world_to_screen(pos, cam_y):
    x, y = pos
    screen_x = int(x * 30)
    screen_y = int((y - cam_y) * -30 + HEIGHT/2)
    return (screen_x, screen_y)

def main():
    model = PPO.load("ppo_lunarlander")

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    obs, info = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("LunarLander - PPO Agent (Camera Fixed)")
    clock = pygame.time.Clock()

    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        frame = env.render()
        lander_y = obs[1]
        cam_y = lander_y - 5

        surf = pygame.surfarray.make_surface(np.rot90(frame))
        screen.blit(surf, (0, 0))
        pygame.display.update()

        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            done = True

        clock.tick(60)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()

