"""Export, evaluate and film a finished training run (CLI twin of notebook Sections 7-9).

    python make_video.py --run-dir runs/quadbot_ppo
"""
import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from quadbot import ENV_ID, QuadBotEnvConfig
from quadbot.policy import NumpyPolicy
from quadbot.training import (
    QuadBotPolicy, evaluate_policy_metrics, evaluation_summary_lines, export_policy_npz, gait_analysis,
)
from quadbot.viz import figure_to_array, footfall_figure, make_showcase_video, training_curve_figure

p = argparse.ArgumentParser()
p.add_argument("--run-dir", default="runs/quadbot_ppo")
p.add_argument("--out", default=None, help="output MP4 (default: <run-dir>/quadbot_demo.mp4)")
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
p.add_argument("--note", default="", help="extra line for the title card")
a = p.parse_args()
run = a.run_dir
out = a.out or os.path.join(run, "quadbot_demo.mp4")
t_start = time.time()

with open(os.path.join(run, "config.json")) as f:
    run_cfg = json.load(f)
yaw_max = max(abs(float(v)) for v in run_cfg["env"]["command_yaw_range"])
env_cfg = QuadBotEnvConfig(command_yaw_range=(-yaw_max, yaw_max))

# 1) Export the final policy and check the NumPy version against SB3 along a rollout.
sb3 = QuadBotPolicy(os.path.join(run, "final_model.zip"), os.path.join(run, "vecnormalize.pkl"))
npz_path = export_policy_npz(sb3.model, sb3.normalizer, os.path.join(run, "quadbot_policy.npz"))
policy = NumpyPolicy(npz_path)
env = gym.make(ENV_ID, config=env_cfg)
obs, _ = env.reset(seed=0, options={"command": (0.7, 0.0, 0.0)})
max_diff = 0.0
for _ in range(300):
    action = policy(obs)
    max_diff = max(max_diff, float(np.abs(action - sb3(obs)).max()))
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        obs, _ = env.reset(options={"command": (0.7, 0.0, 0.0)})
env.close()
print(f"exported {npz_path} ({os.path.getsize(npz_path) / 1e3:.0f} kB), max |NumPy - SB3| = {max_diff:.1e}")

# 2) Evaluate and analyse the gait.
eval_rows = evaluate_policy_metrics(policy, env_cfg, commands=(0.4, 0.7, 1.0), episodes_per_command=2)
for row in eval_rows:
    print(row)
gait = gait_analysis(policy, env_cfg, command=(0.7, 0.0, 0.0), seconds=8.0)
summary = evaluation_summary_lines(eval_rows, gait)
print("\n".join(summary))
print({k: v for k, v in gait.items() if k != "contacts"})

# 3) Figures and the showcase video.
curve_fig = training_curve_figure(run)
foot_fig = footfall_figure(gait["contacts"][:150], gait["dt"],
                           f"Footfall pattern at 0.7 m/s: {gait['label']} (diagonal pairs share a colour)",
                           figsize=(12, 5))
curve_img, foot_img = figure_to_array(curve_fig), figure_to_array(foot_fig)
curve_fig.savefig(os.path.join(run, "learning_curves.png"), facecolor=curve_fig.get_facecolor())
foot_fig.savefig(os.path.join(run, "footfall.png"), facecolor=foot_fig.get_facecolor())
plt.close("all")

subtitle = ["Gymnasium env  +  MuJoCo physics  +  PPO (Stable-Baselines3)",
            f"{run_cfg['total_timesteps'] / 1e6:g}M environment steps" + (f"  -  {a.note}" if a.note else "")]
t0 = time.time()
stats = make_showcase_video(
    policy, env_cfg, out, subtitle_lines=subtitle, curve_image=curve_img,
    extra_cards=[(foot_img, "Gait analysis")], results_lines=summary, gait_name=gait["label"],
    include_turning=yaw_max > 0, width=a.width, height=a.height,
)
print(f"video: {out} ({stats['video']['seconds']:.0f} s, {os.path.getsize(out) / 1e6:.1f} MB), "
      f"rendered in {(time.time() - t0) / 60:.1f} min")
print({k: v for k, v in stats.items() if k != "video"})

report = dict(policy_npz=npz_path, numpy_vs_sb3_max_abs_diff=max_diff, evaluation=eval_rows,
              gait={k: v for k, v in gait.items() if k != "contacts"}, summary=summary, video=stats)
with open(os.path.join(run, "results.json"), "w") as f:
    json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(f"done in {(time.time() - t_start) / 60:.1f} min")
