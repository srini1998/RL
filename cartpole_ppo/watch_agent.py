import gymnasium as gym
from stable_baselines3 import PPO
import time

def main():
    env = gym.make("CartPole-v1", render_mode="human")
    model = PPO.load("ppo_cartpole")

    obs, info = env.reset()
    done = False

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            obs, info = env.reset()
            time.sleep(0.5)

if __name__ == "__main__":
    main()

