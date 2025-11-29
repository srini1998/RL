import gymnasium as gym
from stable_baselines3 import PPO

def main():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50_000)
    model.save("ppo_cartpole")

    env.close()

if __name__ == "__main__":
    main()

