import os
import pickle
import argparse
import json
from experiment import Experiment
from src.environment.environment import Environment
from src.utils.factory import create_env_and_agent


def main(env_config_path, agent_config_path, seed=0, save_dir="results", monitor=False, monitor_after=1000):
    env_config = json.load(open(env_config_path, "r"))
    agent_config = json.load(open(agent_config_path, "r"))
    
    agent_config["parameters"]["seed"] = seed
    env_config["seed"] = seed

    env, agent = create_env_and_agent(
        env_config, agent_config, seed, monitor=monitor, monitor_after=monitor_after
    )
    eval_env = Environment(config=env_config, monitor=monitor, monitor_after=monitor_after)
    
    data = {}
    data["experiment"] = {}

    # Experiment meta-data
    data["experiment"]["environment"] = env_config
    data["experiment"]["agent"] = agent_config


    exp = Experiment(
        agent=agent,
        env=env,
        eval_env=eval_env,
        eval_episodes=env_config["eval_episodes"],
        eval_interval_timesteps=env_config["eval_interval_timesteps"],
        total_timesteps=env_config["total_timesteps"],
    )
    exp.run()

    data["experiment_data"] = {}
    data["experiment_data"]["runs"] = []
    run_data = {}
    run_data["seed"] = seed
    run_data["total_timesteps"] = env_config.get("total_timesteps")
    run_data["eval_interval_timesteps"] = env_config.get("eval_interval_timesteps")
    run_data["episodes_per_eval"] = env_config.get("eval_episodes")

    run_data = {**run_data, **agent.info, **env.info, **exp.info}
    data["experiment_data"]["runs"].append(run_data)

    save_dir = os.path.join(save_dir, f"{env.env_name}/{agent_config['agent_name']}/")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"seed_{seed}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    
    model_save_dir = os.path.join(save_dir, "checkpoints", f"seed_{seed}")
    os.makedirs(model_save_dir, exist_ok=True)
    agent.save_model(
        env_name=env.env_name,
        actor_path=os.path.join(model_save_dir, f"actor.pth"),
        critic_path=os.path.join(model_save_dir, f"critic.pth"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run RL experiment.")
    parser.add_argument(
        "--env_config",
        type=str,
        default="config/environments/PendulumContinuous-v0.json",
        help="Path to the environment config JSON file.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="config/agents/sac.json",
        help="Path to the agent config JSON file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for experiment.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/",
        help="Save directory",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.env_config, args.agent_config, seed=args.seed, save_dir=args.save_dir)