This repository contains the implementation of Soft Actor-Critic (SAC) with modified exploration strategies, including epsilon-greedy and Ornstein-Uhlenbeck (OU) noise action selection.

## Configuration

All agent and environment configuration files are located in the `config/` directory. 
* Environment configurations: `config/environments/`
* Agent configurations: `config/agents/`

## Running the Code

To train an agent, execute `main.py` and provide the paths to your desired environment and agent configuration files, along with the seed and save directory.

```bash
python main.py \
  --env_config <path_to_env_config.json> \
  --agent_config <path_to_agent_config.json> \
  --seed <seed_number> \
  --save_dir <path_to_save_directory>
```

## Generating Plots
To generate comparison plots between different runs, use the provided plotting utility. Pass the directories containing the saved seed results, the display names for the algorithms, the environment name, and the output directory for the generated graphs.

```bash
python src/utils/plot_utils.py \
  --seed_dirs <path_to_baseline_results> <path_to_experiment_results> \
  --algo_names '<Baseline_Name>' '<Experiment_Name>' \
  --env_name <Environment_Name> \
  --save_dir <path_to_save_plots>
```