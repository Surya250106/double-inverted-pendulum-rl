import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment import DoublePendulumEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--reward_type", type=str, default="shaped", choices=["baseline", "shaped"])
    parser.add_argument("--save_path", type=str, default="models/ppo_model.zip")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Environment
    env = DoublePendulumEnv(reward_type=args.reward_type)

    # Wrap with Monitor for logging
    env = Monitor(env, filename="logs/monitor.csv")

    # Model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="logs/"
    )

    # Train
    model.learn(total_timesteps=args.timesteps)

    # Save model
    model.save(args.save_path)

    print(f"Model saved to {args.save_path}")

    env.close()


if __name__ == "__main__":
    main()