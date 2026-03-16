from gymnasium.envs.registration import register

register(
    id="SDNEnv-v0",
    entry_point="rl_agent.envs.sdn_env:SDNEnvironment",
    max_episode_steps=1000,
)
