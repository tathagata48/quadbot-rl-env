"""Command-line training entry point (same code path as the Colab notebook)."""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from quadbot import QuadBotEnvConfig
from quadbot.training import TrainConfig, train

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=6_000_000)
p.add_argument("--envs", type=int, default=16)
p.add_argument("--run-dir", default="runs/quadbot_ppo")
p.add_argument("--yaw", type=float, default=0.0, help="max |yaw-rate command| (rad/s)")
p.add_argument("--dr", action="store_true", help="enable domain randomisation")
p.add_argument("--eval-every", type=int, default=250_000)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--checkpoint-every", type=int, default=500_000)
p.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --run-dir")
a = p.parse_args()

env_cfg = QuadBotEnvConfig(command_yaw_range=(-a.yaw, a.yaw), domain_randomization=a.dr)
cfg = TrainConfig(
    total_timesteps=a.steps, n_envs=a.envs, run_dir=a.run_dir, env=env_cfg,
    eval_every_steps=a.eval_every, seed=a.seed, checkpoint_every_steps=a.checkpoint_every,
)
train(cfg, resume=a.resume)
