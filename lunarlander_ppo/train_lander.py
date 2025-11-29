import gymnasium as gym
from stable_baselines3 import PPO

def main():
    env = gym.make("LunarLander-v2")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.01,
        clip_range=0.2,
    )

    model.learn(total_timesteps=200_000)
    model.save("ppo_lunarlander")

    env.close()
    print("\n🎉 Training finished. Model saved as ppo_lunarlander.zip")

if __name__ == "__main__":
    main()

