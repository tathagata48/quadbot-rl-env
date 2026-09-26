"""Assemble QuadBot_RL_Colab.ipynb from the package sources (single source of truth)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
PKG = ROOT / "quadbot"
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


def module(name: str) -> None:
    src = (PKG / name).read_text()
    code(f"%%writefile quadbot/{name}\n{src}")


md(r"""
# QuadBot — a custom quadruped that learns to walk
### Gymnasium environment · MuJoCo physics · PPO (Stable-Baselines3)

This notebook builds a quadruped robot **from scratch**, wraps it in a **Gymnasium** environment (`QuadBot-v0`), trains a velocity-tracking locomotion policy with **PPO**, evaluates it and analyses its gait, exports it to a torch-free NumPy policy, renders an MP4, and lets you watch and steer the running simulation in an interactive 3D viewer.

**Pipeline:** MJCF generator → `QuadBot-v0` → PPO (16 envs + `VecNormalize`) → evaluation → `.npz` policy export → video → interactive 3D simulation viewer

| Robot | |
|---|---|
| Degrees of freedom | 12 — abduction, hip, knee on each of 4 legs |
| Mass / standing height | 11.3 kg / 0.30 m |
| Leg segments | thigh 0.20 m, calf 0.20 m, spherical rubber feet |
| Actuation | joint PD servos at 250 Hz, kp = 50 N·m/rad, kd = 1.5 N·m·s/rad, 25–35 N·m torque limits |

| Task: `QuadBot-v0` | |
|---|---|
| Action (12) | joint-angle offsets from the standing pose, ±0.3 rad, at 50 Hz |
| Observation (48) | gravity vector, base linear/angular velocity, command, joint positions/velocities, previous action (body frame) |
| Command | forward speed 0.3–1.0 m/s and yaw rate ±0.6 rad/s, resampled every 10 s |
| Reward | velocity tracking + regularisers: orientation, height, torque, smoothness, foot air-time, foot slip, collisions, joint limits |
| Episode | 20 s; terminates if the torso touches the ground, drops below 0.16 m, or tilts past 60° |

**Runtime.** A normal **CPU runtime is enough** — this MLP policy trains faster on CPU than on a GPU. The full 8 M-step run takes roughly **an hour** on Colab. Set `QUICK_RUN = True` in Section 6 for a ~2-minute smoke test, or skip to **Section 11** to load the pretrained policy (`quadbot_policy.npz`) that comes with this notebook, then watch it walk in **Section 12**.
""")

md("## 1 · Setup\nInstalls the packages and picks a headless OpenGL backend for MuJoCo (NVIDIA EGL on GPU runtimes, Mesa EGL or OSMesa on CPU runtimes). `MUJOCO_GL` must be set before MuJoCo is imported, so run this cell first.")
code(r'''
# @title Install dependencies and configure headless rendering
import importlib.util
import json
import os
import shutil
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
PROJECT_DIR = "/content/quadbot_project" if IN_COLAB else os.path.abspath("quadbot_project")
os.makedirs(os.path.join(PROJECT_DIR, "quadbot"), exist_ok=True)
os.chdir(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

REQUIRED = {  # import name -> pip requirement
    "mujoco": "mujoco>=3.2",
    "gymnasium": "gymnasium>=1.0",
    "stable_baselines3": "stable-baselines3>=2.4",
    "imageio_ffmpeg": "imageio[ffmpeg]",
    "tensorboard": "tensorboard",
    "pandas": "pandas",
}
missing = [req for mod, req in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

PROBE = (
    "import mujoco\n"
    "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom size=\".1\"/></worldbody></mujoco>')\n"
    "r = mujoco.Renderer(m, 32, 32); r.update_scene(mujoco.MjData(m)); r.render(); print('RENDER_OK')"
)


def render_backend_works(backend: str) -> bool:
    env = dict(os.environ, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
    try:
        out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return False
    return "RENDER_OK" in out.stdout


if shutil.which("nvidia-smi") and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0:
    # Colab installs the NVIDIA driver without the EGL vendor file that glvnd needs.
    icd = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not os.path.exists(icd) and os.access(os.path.dirname(icd) or "/", os.W_OK):
        with open(icd, "w") as f:
            json.dump({"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}, f)

MUJOCO_BACKEND = "egl" if render_backend_works("egl") else None
if MUJOCO_BACKEND is None and shutil.which("apt-get"):
    print("EGL unavailable -> installing OSMesa (software rendering)...")
    subprocess.run("apt-get -qq update && apt-get -qq install -y libosmesa6-dev > /dev/null", shell=True)
    MUJOCO_BACKEND = "osmesa" if render_backend_works("osmesa") else None
if MUJOCO_BACKEND is None:
    raise RuntimeError("No headless OpenGL backend found; install libegl1 or libosmesa6.")
os.environ["MUJOCO_GL"] = MUJOCO_BACKEND
os.environ["PYOPENGL_PLATFORM"] = MUJOCO_BACKEND
print(f"Rendering backend: {MUJOCO_BACKEND} | CPU cores: {os.cpu_count()} | project dir: {PROJECT_DIR}")
''')

md(r"""
## 2 · The robot (MJCF generator)
The robot is described by a `RobotSpec` dataclass and turned into MJCF by `build_quadbot_xml()`. Generating the XML keeps the four legs consistent and makes design studies easy (change a leg length or motor gain in Python and rebuild).

* **Kinematics per leg:** torso → *abad* (x-axis) → hip link → *hip* (y-axis) → thigh → *knee* (y-axis) → calf → foot.
* **Collisions:** robot geoms only collide with the ground (never with each other); decorative geoms are visual-only and massless.
* **Motors:** MuJoCo `<position>` actuators = joint-level PD loops with torque limits, integrated implicitly for stability.
* **Sensors:** a touch sensor on each foot (used for the gait reward and the video HUD).

The package is written to disk with `%%writefile` so that worker processes (used automatically on machines with ≥ 4 cores) can import it.
""")
module("robot.py")

md(r"""
## 3 · The Gymnasium environment
`QuadBotEnv` subclasses `gymnasium.Env` directly (no dependence on Gymnasium's internal MuJoCo base class), so it is stable across Gymnasium versions. Highlights:

* **Observation (48-D, body frame):** projected gravity, base linear velocity (privileged in simulation — swap for a state estimator on hardware via `include_base_lin_vel=False`), base angular velocity, command, joint positions relative to the standing pose, joint velocities, previous action.
* **Reward (`DEFAULT_REWARD_SCALES`):** exponential velocity-tracking kernels plus legged-gym-style regularisers. The per-step total is clipped at zero (`only_positive_rewards`) so penalties never make early termination attractive.
* **Domain randomisation (optional):** ground/foot friction, payload mass, motor strength and random pushes — the usual ingredients for sim-to-real transfer.
* `info` contains only floats (cheap to copy in vectorised envs); foot contacts are exposed as `env.unwrapped.foot_contacts`.
""")
module("env.py")
module("policy.py")
module("__init__.py")

md(r"""
## 4 · Training and video utilities
* `training.py` — vectorised env construction, PPO configuration, a callback that logs forward speed / tracking error / throughput, checkpointing and evaluation, and `export_policy_npz()`.
* `viz.py` — tracking-camera rollouts with a telemetry HUD and foot-contact lights, title cards, learning-curve plots, and the showcase-video composer.
""")
module("training.py")
module("viz.py")

md("## 5 · Inspect the robot and validate the environment")
code(r'''
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
from IPython.display import Video, display

import quadbot
from quadbot import ENV_ID, QuadBotEnvConfig, RobotSpec
from quadbot.policy import NumpyPolicy
from quadbot.training import (
    TrainConfig, evaluate_policy_metrics, evaluation_summary_lines, export_policy_npz, gait_analysis, train,
)
from quadbot.viz import figure_to_array, footfall_figure, make_showcase_video, record_rollout, training_curve_figure

spec = RobotSpec()
env = gym.make(ENV_ID, render_mode="rgb_array")
base = env.unwrapped
print(f"{ENV_ID}: obs {env.observation_space.shape}, action {env.action_space.shape}, "
      f"control dt {base.dt * 1000:.0f} ms, physics dt {base.model.opt.timestep * 1000:.0f} ms")
print(f"Robot mass {spec.total_mass:.2f} kg | nominal standing height {spec.nominal_height:.3f} m | "
      f"bodies {base.model.nbody} | actuators {base.model.nu}")

env.reset(seed=0)
views = [("front-left", 140, -15, 1.4), ("side", 90, -5, 1.2), ("top", 180, -60, 1.5)]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (name, az, el, dist) in zip(axes, views):
    base._renderer = None  # rebuild the camera for each view
    base.cfg.camera_azimuth, base.cfg.camera_elevation, base.cfg.camera_distance = az, el, dist
    ax.imshow(base.render_frame(640, 400))
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.show()
env.close()
''')
code(r'''
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

gym_check_env(gym.make(ENV_ID).unwrapped, skip_render_check=True)
sb3_check_env(gym.make(ENV_ID).unwrapped, warn=False)
print("Gymnasium and Stable-Baselines3 API checks passed")

# Zero action = hold the standing pose: the robot must stay up.
env = gym.make(ENV_ID)
obs, info = env.reset(seed=1)
for t in range(250):
    obs, reward, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
    assert not terminated, "robot fell while holding the standing pose"
print(f"Standing test passed: height {info['base_height']:.3f} m after {250 * base.dt:.0f} s")

# Raw simulation throughput with random actions.
t0, steps = time.perf_counter(), 3000
obs, _ = env.reset(seed=2)
for _ in range(steps):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, _ = env.reset()
print(f"Single-env speed: {steps / (time.perf_counter() - t0):,.0f} steps/s")
env.close()
''')

md(r"""
## 6 · Train PPO
PPO settings (in `TrainConfig`): 16 environments × 256 steps = 4,096 samples per update, 5 epochs with minibatches of 1,024, γ = 0.99, λ = 0.95, clip 0.2, learning rate 3e-4 → 5e-5 (linear), ELU MLP (256, 256, 128) for both actor and critic, observation and reward normalisation.

Progress is printed every 100k steps. Checkpoints (with normaliser stats) are written every 500k steps and the best evaluation model is kept in `runs/quadbot_ppo/best/` together with its matching normaliser.

**Interrupted?** With `RESUME = True`, re-running the training cell continues from the newest checkpoint (same learning-rate schedule, learning curve and evaluation log). Colab wipes `/content` when the runtime is recycled, so for long runs point `RUN_DIR` at a mounted Google Drive folder.
""")
code(r'''
# @title Training configuration { run: "auto" }
QUICK_RUN = False  # @param {type:"boolean"}
TOTAL_TIMESTEPS = 8_000_000  # @param {type:"integer"}
TURNING_COMMANDS = True  # @param {type:"boolean"}
DOMAIN_RANDOMIZATION = False  # @param {type:"boolean"}
N_ENVS = 16  # @param {type:"integer"}
RESUME = True  # @param {type:"boolean"}
RUN_DIR = "runs/quadbot_ppo"
VIDEO_RESOLUTION = (1280, 720)

env_cfg = QuadBotEnvConfig(
    command_yaw_range=(-0.6, 0.6) if TURNING_COMMANDS else (0.0, 0.0),
    domain_randomization=DOMAIN_RANDOMIZATION,
)
train_cfg = TrainConfig(
    total_timesteps=150_000 if QUICK_RUN else TOTAL_TIMESTEPS,
    n_envs=N_ENVS,
    run_dir=RUN_DIR,
    env=env_cfg,
    eval_every_steps=50_000 if QUICK_RUN else 250_000,
    checkpoint_every_steps=0 if QUICK_RUN else 500_000,
)
print(f"{train_cfg.total_timesteps:,} steps with {N_ENVS} envs "
      f"(expect roughly {train_cfg.total_timesteps / 2500 / 60:.0f} min at ~2,500 steps/s)")
''')
code(r'''
model, vecnorm, metrics = train(train_cfg, resume=RESUME)
''')
code(r'''
# Optional: live TensorBoard (run in a separate cell while training, or afterwards).
%load_ext tensorboard
%tensorboard --logdir runs
''')

md("## 7 · Learning curves")
code(r'''
fig = training_curve_figure(RUN_DIR)
plt.show()
''')

md(r"""
## 8 · Export, evaluate, and analyse the gait
The actor network and the observation normaliser are exported to a single `.npz`. `NumpyPolicy` runs it with NumPy only (verified to match SB3's deterministic actions to ~1e-7), which is handy for deployment and makes the file independent of SB3/PyTorch versions.

`gait_analysis()` then records foot contacts at 0.7 m/s. Two feet are *in phase* when both are on the ground or both are in the air: diagonal pairs in phase means a **trot**, same-side pairs a **pace**, front/rear pairs a **bound**.
""")
code(r'''
POLICY_PATH = export_policy_npz(model, vecnorm, f"{RUN_DIR}/quadbot_policy.npz")
policy = NumpyPolicy(POLICY_PATH)

# Sanity check: NumPy policy == SB3 policy.
probe_env = gym.make(ENV_ID, config=env_cfg)
obs, _ = probe_env.reset(seed=0)
sb3_action, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
print("max |SB3 - NumPy| action difference:", float(np.abs(sb3_action - policy(obs)).max()))
probe_env.close()

eval_rows = evaluate_policy_metrics(policy, env_cfg, commands=(0.4, 0.7, 1.0), episodes_per_command=2)
results = pd.DataFrame(eval_rows)
display(results)
print(f"Falls: {int(results.fell.sum())}/{len(results)} | "
      f"mean |vx error|: {np.mean(np.abs(results.mean_vx - results.command_vx)):.3f} m/s")

gait = gait_analysis(policy, env_cfg, command=(0.7, 0.0, 0.0), seconds=8.0)
print(f"Gait at 0.7 m/s: {gait['label']} | in phase: diagonal {gait['diagonal_sync']:.0%}, "
      f"same side {gait['lateral_sync']:.0%}, front/rear {gait['fore_hind_sync']:.0%} | "
      f"{gait['stride_hz']:.2f} strides/s | duty factor {np.mean(gait['duty_factor']):.2f}")
fig = footfall_figure(gait["contacts"][:150], gait["dt"],
                      f"Footfall pattern at 0.7 m/s: {gait['label']} (diagonal pairs share a colour)")
plt.show()
''')

md(r"""
## 9 · Record the video
The showcase video contains: a title card, the untrained robot (random actions), the trained policy following forward-speed commands (0.4 → 0.7 → 1.0 m/s), turning on yaw-rate commands, a 5× slow-motion close-up of the gait (every 4 ms physics step is rendered, so the slow motion stays smooth), the learning curve, the footfall diagram and a results card. The HUD shows commanded vs. measured velocity and lights up each foot while it is in contact.

Frames are streamed straight into the H.264 encoder, so memory use stays flat. CPU runtimes render with software OpenGL: 720p takes several minutes, so lower `VIDEO_RESOLUTION` (Section 6) for a quicker preview.
""")
code(r'''
VIDEO_PATH = f"{RUN_DIR}/quadbot_demo.mp4"
curve_fig = training_curve_figure(RUN_DIR)
foot_fig = footfall_figure(gait["contacts"][:150], gait["dt"],
                           f"Footfall pattern at 0.7 m/s: {gait['label']} (diagonal pairs share a colour)",
                           figsize=(12, 5))
curve_img, foot_img = figure_to_array(curve_fig), figure_to_array(foot_fig)
plt.close(curve_fig)
plt.close(foot_fig)

t0 = time.time()
video_stats = make_showcase_video(
    policy, env_cfg, VIDEO_PATH,
    subtitle_lines=[
        "Gymnasium env  +  MuJoCo physics  +  PPO (Stable-Baselines3)",
        f"{train_cfg.total_timesteps / 1e6:g}M environment steps",
    ],
    curve_image=curve_img,
    extra_cards=[(foot_img, "Gait analysis")],
    results_lines=evaluation_summary_lines(eval_rows, gait),
    gait_name=gait["label"],
    include_turning=TURNING_COMMANDS,
    width=VIDEO_RESOLUTION[0], height=VIDEO_RESOLUTION[1],
)
print(f"Rendered {video_stats['video']['seconds']:.0f} s of video in {(time.time() - t0) / 60:.1f} min")
print({k: v for k, v in video_stats.items() if k != "video"})
display(Video(VIDEO_PATH, embed=True, width=960))
''')

md("## 10 · Save and download")
code(r'''
import zipfile

ARCHIVE = "quadbot_results.zip"
keep = [
    f"{RUN_DIR}/final_model.zip", f"{RUN_DIR}/vecnormalize.pkl", f"{RUN_DIR}/quadbot_policy.npz",
    f"{RUN_DIR}/config.json", f"{RUN_DIR}/training_history.json", f"{RUN_DIR}/eval/evaluations.npz",
    VIDEO_PATH,
] + [f"quadbot/{name}" for name in sorted(os.listdir("quadbot")) if name.endswith(".py")]
with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in keep:
        if os.path.exists(path):
            zf.write(path)
print(f"{ARCHIVE}: {os.path.getsize(ARCHIVE) / 1e6:.1f} MB")

if IN_COLAB:
    from google.colab import files
    files.download(ARCHIVE)
    # To keep results across sessions instead:
    # from google.colab import drive; drive.mount("/content/drive")
    # shutil.copy(ARCHIVE, "/content/drive/MyDrive/")
''')

md(r"""
## 11 · (Optional) Skip training — use the pretrained policy
Run Sections 1–5, then this cell. Upload **`quadbot_policy.npz`** when prompted (outside Colab, put it in the project folder). The pretrained policy expects the default `QuadBotEnvConfig` and `RobotSpec` (48-D observation with base velocity). The cell evaluates the policy, plots its footfall pattern and renders the showcase video (several minutes at 720p on a CPU runtime).
""")
code(r'''
PRETRAINED_NPZ = "quadbot_policy.npz"
if IN_COLAB and not os.path.exists(PRETRAINED_NPZ):
    from google.colab import files
    uploaded = files.upload()
    PRETRAINED_NPZ = next(name for name in uploaded if name.endswith(".npz"))

pretrained = NumpyPolicy(PRETRAINED_NPZ)
pre_cfg = QuadBotEnvConfig(command_yaw_range=(-0.6, 0.6))
pre_rows = evaluate_policy_metrics(pretrained, pre_cfg, commands=(0.4, 0.7, 1.0), episodes_per_command=1)
display(pd.DataFrame(pre_rows))
pre_gait = gait_analysis(pretrained, pre_cfg, command=(0.7, 0.0, 0.0), seconds=8.0)
fig = footfall_figure(pre_gait["contacts"][:150], pre_gait["dt"],
                      f"Pretrained policy at 0.7 m/s: {pre_gait['label']} (diagonal pairs share a colour)")
plt.show()

res = globals().get("VIDEO_RESOLUTION", (1280, 720))
pre_stats = make_showcase_video(
    pretrained, pre_cfg, "quadbot_pretrained_demo.mp4",
    subtitle_lines=["Pretrained policy (8M PPO steps)", "Gymnasium  +  MuJoCo  +  Stable-Baselines3"],
    results_lines=evaluation_summary_lines(pre_rows, pre_gait),
    gait_name=pre_gait["label"],
    width=res[0], height=res[1],
)
display(Video("quadbot_pretrained_demo.mp4", embed=True, width=960))
''')

md(r"""
## 12 · Watch the simulation itself (interactive 3D)
The video in Section 9 is pre-rendered. This viewer shows the simulation while it runs: MuJoCo and the policy step in this notebook's Python kernel, and your browser draws every body at the pose MuJoCo computed (three.js / WebGL), so you can orbit, zoom and pan freely.

* **Live (Colab):** the physics advances in real time while you steer with the *speed* and *turn* sliders or the arrow keys (click the view first). **push** (space) shoves the robot sideways and **reset** (R) starts over. Commands are limited to the ranges the policy was trained on. The view pauses while another cell is running, because the kernel is what steps the simulation.
* **Replay (any notebook):** a 22 s tour of speed and turning commands is simulated first, then played back with pause, scrubbing and slow motion. `REPLAY_PUSH_EVERY_S` adds a sideways shove every few seconds.

Feet light up while they touch the ground, the orange line traces the torso's path, and red arrows mark pushes. The viewer loads three.js from cdn.jsdelivr.net. For a replay page that opens in any browser, run `sim_viewer.save_simulation_html("quadbot_sim.html", view_policy, view_cfg)`.

The first cell writes `quadbot/sim_viewer.py`. The second picks a policy (the pretrained one from Section 11, else the one trained in Section 8, else it asks for `quadbot_policy.npz`) and opens the viewer.
""")
module("sim_viewer.py")
code('''
# @title Interactive 3D simulation
VIEW_MODE = "live (Colab)"  # @param ["live (Colab)", "replay"]
REPLAY_PUSH_EVERY_S = 0  # @param {type:"slider", min:0, max:10, step:1}
VIEW_HEIGHT_PX = 560  # @param {type:"slider", min:360, max:900, step:20}

import importlib
import quadbot.sim_viewer as sim_viewer
importlib.reload(sim_viewer)

if globals().get("pretrained") is not None:  # Section 11
    view_policy, view_cfg = pretrained, pre_cfg
elif globals().get("policy") is not None:  # Sections 6-8
    view_policy, view_cfg = policy, env_cfg
else:
    npz_path = "quadbot_policy.npz"
    if IN_COLAB and not os.path.exists(npz_path):
        from google.colab import files
        npz_path = next(name for name in files.upload() if name.endswith(".npz"))
    view_policy, view_cfg = NumpyPolicy(npz_path), QuadBotEnvConfig(command_yaw_range=(-0.6, 0.6))

view_mode = "live" if VIEW_MODE.startswith("live") and sim_viewer.in_colab() else "replay"
print(f"Showing the {view_mode} simulation")
sim_viewer.show_simulation(view_policy, view_cfg, mode=view_mode,
                           push_every=REPLAY_PUSH_EVERY_S, height=VIEW_HEIGHT_PX)
''')

md(r"""
## 13 · Where to take it next
* **Sim-to-real:** set `DOMAIN_RANDOMIZATION = True`, train with `include_base_lin_vel=False` (or add a learned velocity estimator), add `obs_noise_scale`, and model actuator latency.
* **Rough terrain:** add a height-field to the MJCF (collision `contype=0 conaffinity=1`) and a curriculum over its roughness.
* **Richer commands:** widen `command_vy_range` for lateral walking, add heading commands, or a stand-still command with a `stand_still` penalty.
* **Gait shaping:** `gait_analysis()` quantifies the gait that emerges; add a phase-clock observation and a contact-schedule reward to request a specific gait (trot, pace, bound) or step frequency.
* **Deployment:** `quadbot_policy.npz` holds the complete actor; `NumpyPolicy` is ~40 lines and runs anywhere NumPy does (or convert the same weights to ONNX).
* **Local viewer:** `gym.make("QuadBot-v0", render_mode="human")` opens MuJoCo's interactive viewer on a desktop machine.
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "colab": {"provenance": [], "toc_visible": True},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
out = ROOT / "QuadBot_RL_Colab.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
