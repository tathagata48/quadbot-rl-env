"""Steer the trained QuadBot policy in MuJoCo's own interactive window.

Run this locally (it needs a real display):

    export MUJOCO_GL=glfw
    python3 teleop.py --policy runs/main/quadbot_policy.npz

Controls (click the MuJoCo window first so it has keyboard focus):
    Up / Down     increase / decrease forward speed
    Left / Right  turn left / right
    Space         zero the command (stand/stop)
    P             shove the robot sideways (a random push, like the training pushes)
    R             reset the episode
    Esc           quit

The on-screen overlay is MuJoCo's own; the command and measured speed also print to the
terminal a few times a second so you can watch them without the overlay.
"""
from __future__ import annotations

import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from quadbot import QuadBotEnvConfig
from quadbot.env import QuadBotEnv
from quadbot.policy import NumpyPolicy

VX_STEP = 0.05
YAW_STEP = 0.1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default="runs/main/quadbot_policy.npz", help="path to the .npz policy")
    parser.add_argument("--vx", type=float, default=0.6, help="starting forward speed (m/s)")
    parser.add_argument("--yaw-limit", type=float, default=0.6, help="max turn rate offered (rad/s)")
    parser.add_argument("--push-speed", type=float, default=0.5, help="sideways kick speed for the P key (m/s)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = QuadBotEnvConfig(command_yaw_range=(-args.yaw_limit, args.yaw_limit))
    env = QuadBotEnv(config=cfg)
    policy = NumpyPolicy(args.policy)
    vx_lo, vx_hi = cfg.command_vx_range
    rng = np.random.default_rng(args.seed)

    state = {"vx": float(np.clip(args.vx, vx_lo, vx_hi)), "yaw": 0.0, "push": 0.0, "reset": False, "quit": False}

    def key_callback(keycode: int) -> None:
        # GLFW key codes: arrows, space, letters (as uppercase ASCII), Esc.
        if keycode == 265:  # Up
            state["vx"] = float(np.clip(state["vx"] + VX_STEP, vx_lo, vx_hi))
        elif keycode == 264:  # Down
            state["vx"] = float(np.clip(state["vx"] - VX_STEP, vx_lo, vx_hi))
        elif keycode == 263:  # Left
            state["yaw"] = float(np.clip(state["yaw"] + YAW_STEP, -args.yaw_limit, args.yaw_limit))
        elif keycode == 262:  # Right
            state["yaw"] = float(np.clip(state["yaw"] - YAW_STEP, -args.yaw_limit, args.yaw_limit))
        elif keycode == 32:  # Space
            state["vx"], state["yaw"] = 0.0, 0.0
        elif keycode == ord("P"):
            state["push"] = args.push_speed
        elif keycode == ord("R"):
            state["reset"] = True
        elif keycode == 256:  # Esc
            state["quit"] = True

    obs, _ = env.reset(seed=args.seed, options={"command": (state["vx"], 0.0, state["yaw"])})
    env.set_command(state["vx"], 0.0, state["yaw"])
    falls = 0
    last_print = 0.0
    print(
        f"QuadBot teleop — vx range [{vx_lo:.2f}, {vx_hi:.2f}] m/s, yaw limit ±{args.yaw_limit:.2f} rad/s\n"
        "Click the MuJoCo window, then: Up/Down speed, Left/Right turn, Space stop, P push, R reset, Esc quit."
    )
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        viewer.cam.trackbodyid = env._torso_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance = 1.6
        viewer.cam.elevation = -20
        while viewer.is_running() and not state["quit"]:
            step_start = time.perf_counter()
            if state["reset"]:
                obs, _ = env.reset(options={"command": (state["vx"], 0.0, state["yaw"])})
                state["reset"] = False
            env.set_command(state["vx"], 0.0, state["yaw"])
            if state["push"]:
                side = rng.choice((-1.0, 1.0))
                heading = env.data.qpos[3:7]
                yaw = np.arctan2(
                    2.0 * (heading[0] * heading[3] + heading[1] * heading[2]),
                    1.0 - 2.0 * (heading[2] ** 2 + heading[3] ** 2),
                )
                env.data.qvel[0:2] += state["push"] * side * np.array([-np.sin(yaw), np.cos(yaw)])
                state["push"] = 0.0
            obs, _, terminated, _, info = env.step(policy(obs))
            if terminated:
                falls += 1
                print(f"fell (falls so far: {falls}) — resetting")
                obs, _ = env.reset(options={"command": (state["vx"], 0.0, state["yaw"])})
            viewer.sync()
            now = time.perf_counter()
            if now - last_print > 0.3:
                last_print = now
                print(
                    f"\rcommand vx {state['vx']:+.2f} yaw {state['yaw']:+.2f} rad/s | "
                    f"measured vx {info['x_velocity']:+.2f} yaw {info['yaw_rate']:+.2f} rad/s | "
                    f"height {info['base_height']:.3f} m | falls {falls}   ",
                    end="", flush=True,
                )
            time.sleep(max(0.0, env.dt - (now - step_start)))
    env.close()
    print("\nquit")


if __name__ == "__main__":
    sys.exit(main())
