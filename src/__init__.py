from gymnasium.envs.registration import register

register(
    id="Dice421-v0",
    entry_point="src.Dice421:Dice421Env",
)
