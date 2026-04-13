import os
import pandas as pd
import matplotlib.pyplot as plt


def load_log(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Log file not found: {path}")

    df = pd.read_csv(path, comment="#")

    # Stable-Baselines3 Monitor format
    if "l" in df.columns and "r" in df.columns:
        # cumulative timesteps
        df["timesteps"] = df["l"].cumsum()
        df["mean_reward"] = df["r"]
    else:
        raise ValueError("Unexpected log format")

    return df


def main():
    baseline_log = "logs/baseline_monitor.csv"
    shaped_log = "logs/shaped_monitor.csv"

    df_baseline = load_log(baseline_log)
    df_shaped = load_log(shaped_log)

    plt.figure()

    plt.plot(df_baseline["timesteps"], df_baseline["mean_reward"], label="Baseline Reward")
    plt.plot(df_shaped["timesteps"], df_shaped["mean_reward"], label="Shaped Reward")

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Reward Comparison: Baseline vs Shaped")
    plt.legend()

    plt.savefig("reward_comparison.png")
    plt.close()

    print("Saved reward_comparison.png")


if __name__ == "__main__":
    main()