from src.environment import Environment
from src.agent import (
    SAC, 
    EpsilonGreedySAC,
    OUNoiseSAC,
)


AGENT_NAME_TO_CLASS_MAP = {
    "SAC": SAC,
    "EpsilonGreedySAC" : EpsilonGreedySAC,
    "OUNoiseSAC": OUNoiseSAC,
}

def create_env_and_agent(env_config, agent_config, seed, monitor=False, monitor_after=1000):
    agent_config["parameters"]["seed"] = seed
    env_config["seed"] = seed

    env = Environment(config=env_config, monitor=monitor, monitor_after=monitor_after)
    agent_name = agent_config["agent_name"]
    if agent_name not in AGENT_NAME_TO_CLASS_MAP:
        raise ValueError(f"Agent name {agent_name} not recognized. Available agents: {list(AGENT_NAME_TO_CLASS_MAP.keys())}")
    agent_class = AGENT_NAME_TO_CLASS_MAP[agent_name]
    agent = agent_class(**agent_config["parameters"], env=env)
    return env, agent
    