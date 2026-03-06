import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


ALGO_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
]


def load_seed_data(path):

    with open(path, "rb") as f:
        data = pickle.load(f)

    run = data["experiment_data"]["runs"][0]

    eval_rewards = run["eval_episode_rewards"]
    eval_timesteps = np.array(run["timesteps_at_eval"])

    eval_mean_rewards = np.array([np.mean(r) for r in eval_rewards])

    train_rewards = np.array(run["train_episode_rewards"])
    train_steps = np.array(run["train_episode_steps"])

    return {
        "eval_rewards": eval_mean_rewards,
        "eval_timesteps": eval_timesteps,
        "train_rewards": train_rewards,
        "train_steps": train_steps,
    }


def compute_train_timesteps(train_steps):
    return np.cumsum(train_steps)


def load_algorithm_seeds(seed_dir):

    seed_files = [
        os.path.join(seed_dir, f)
        for f in os.listdir(seed_dir)
        if f.endswith(".pkl")
    ]

    return [load_seed_data(f) for f in seed_files]


# ------------------------
# Evaluation Plot
# ------------------------

def plot_eval_curves(algo_data, algo_names, save_path, title: str):

    plt.figure(figsize=(8,5))

    for i, (algo, seeds) in enumerate(algo_data.items()):

        color = ALGO_COLORS[i % len(ALGO_COLORS)]

        min_len = min(len(s["eval_rewards"]) for s in seeds)
        xs = seeds[0]["eval_timesteps"][:min_len]

        curves = np.array([s["eval_rewards"][:min_len] for s in seeds])
        mean_curve = curves.mean(axis=0)

        # seed curves
        for c in curves:
            plt.plot(xs, c, color=color, alpha=0.3, linewidth=1)

        # mean curve
        plt.plot(xs, mean_curve, color=color, linewidth=1, label=algo)

    plt.xlabel("Training Timesteps")
    plt.ylabel("Evaluation Return")
    plt.title(title if title else "Offline Evaluation Rollout Performance")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "eval_returns.png"))
    plt.close()


# ------------------------
# Training Plot
# ------------------------

def plot_train_curves(algo_data, save_path, title: str):

    plt.figure(figsize=(8,5))

    for i, (algo, seeds) in enumerate(algo_data.items()):

        color = ALGO_COLORS[i % len(ALGO_COLORS)]

        train_ts = [compute_train_timesteps(s["train_steps"]) for s in seeds]

        max_t = min(ts[-1] for ts in train_ts)
        grid = np.linspace(0, max_t, 1000)

        curves = []

        for s, ts in zip(seeds, train_ts):
            interp = np.interp(grid, ts, s["train_rewards"])
            curves.append(interp)

        curves = np.array(curves)
        mean_curve = curves.mean(axis=0)

        # seed curves
        for c in curves:
            plt.plot(grid, c, color=color, alpha=0.3, linewidth=1)

        # mean curve
        plt.plot(grid, mean_curve, color=color, linewidth=1, label=algo)

    plt.xlabel("Training Timesteps")
    plt.ylabel("Episode Return")
    plt.title(title if title else "Online Performance")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "train_returns.png"))
    plt.close()


# ------------------------
# Main
# ------------------------

def main(seed_dirs, algo_names, env_name, save_dir):

    save_path = os.path.join(save_dir, env_name)
    os.makedirs(save_path, exist_ok=True)

    algo_data = {}

    for algo, directory in zip(algo_names, seed_dirs):
        algo_data[algo] = load_algorithm_seeds(directory)

    plot_eval_curves(algo_data, algo_names, save_path, title=f"{env_name} Evaluation Performance")
    plot_train_curves(algo_data, save_path, title=f"{env_name} Online Performance")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed_dirs",
        nargs="+",
        required=True,
        help="List of directories containing seed results (one per algorithm)"
    )

    parser.add_argument(
        "--algo_names",
        nargs="+",
        required=True,
        help="Algorithm names corresponding to seed_dirs"
    )

    parser.add_argument(
        "--env_name",
        required=True,
        help="Environment name used as subdirectory"
    )

    parser.add_argument(
        "--save_dir",
        default="plots",
        help="Directory to save plots"
    )

    args = parser.parse_args()

    assert len(args.seed_dirs) == len(args.algo_names), \
        "seed_dirs and algo_names must match"

    main(args.seed_dirs, args.algo_names, args.env_name, args.save_dir)