import argparse
import os
import pygame
import imageio

from stable_baselines3 import PPO
from environment import DoublePendulumEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/ppo_model.zip")
    parser.add_argument("--gif_path", type=str, default="media/agent_final.gif")
    parser.add_argument("--steps", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure media directory exists
    os.makedirs("media", exist_ok=True)

    # Load environment
    env = DoublePendulumEnv(reward_type="shaped")

    # Load trained model
    model = PPO.load(args.model_path, env=env)

    obs, _ = env.reset()

    frames = []

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

        env.render()

        # Capture frame for GIF
        frame = pygame.surfarray.array3d(env.screen)
        frame = frame.transpose([1, 0, 2])  # Convert to (H, W, C)
        frames.append(frame)

        if done:
            obs, _ = env.reset()

    env.close()

    # Save GIF
    imageio.mimsave(args.gif_path, frames, fps=30)

    print(f"GIF saved to {args.gif_path}")


if __name__ == "__main__":
    main()