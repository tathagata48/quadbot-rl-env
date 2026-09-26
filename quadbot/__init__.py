"""QuadBot: a custom quadruped robot + Gymnasium locomotion task."""
from gymnasium.envs.registration import register, registry

from .env import DEFAULT_REWARD_SCALES, QuadBotEnv, QuadBotEnvConfig
from .robot import JOINT_NAMES, LEG_NAMES, RobotSpec, build_quadbot_xml

ENV_ID = "QuadBot-v0"
MAX_EPISODE_STEPS = 1000  # 20 s at 50 Hz

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="quadbot.env:QuadBotEnv", max_episode_steps=MAX_EPISODE_STEPS)

__all__ = [
    "DEFAULT_REWARD_SCALES", "ENV_ID", "JOINT_NAMES", "LEG_NAMES", "MAX_EPISODE_STEPS",
    "QuadBotEnv", "QuadBotEnvConfig", "RobotSpec", "build_quadbot_xml",
]
