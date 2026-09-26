"""Rendering helpers: rollouts to frames, HUD overlays, title cards, MP4 export."""
from __future__ import annotations

import os
from typing import Callable, Sequence

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import ENV_ID, LEG_NAMES
from .env import QuadBotEnvConfig

_ORANGE = (245, 133, 31)
_CYAN = (26, 217, 255)
_WHITE = (240, 244, 248)
_GREY = (140, 150, 160)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw_hud(
    frame: np.ndarray,
    title: str,
    lines: Sequence[str],
    contacts: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay a translucent telemetry panel (and optional foot-contact lights)."""
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale = img.size[1] / 720
    f_title, f_body = _font(int(30 * scale)), _font(int(22 * scale))
    pad = int(18 * scale)
    line_h = int(32 * scale)
    text_w = max([draw.textlength(title, font=f_title)] + [draw.textlength(s, font=f_body) for s in lines])
    width = int(text_w + 2 * pad)
    height = pad * 2 + int(40 * scale) + line_h * len(lines)
    draw.rounded_rectangle((pad, pad, pad + width, pad + height), radius=int(14 * scale), fill=(8, 12, 20, 170))
    draw.text((2 * pad, 1.6 * pad), title, font=f_title, fill=_ORANGE)
    for i, line in enumerate(lines):
        draw.text((2 * pad, 1.6 * pad + int(44 * scale) + i * line_h), line, font=f_body, fill=_WHITE)

    if contacts is not None:
        box_w, box_h = int(64 * scale), int(34 * scale)
        x0 = pad
        y0 = img.size[1] - pad - 2 * box_h - int(46 * scale)
        draw.rounded_rectangle(
            (x0, y0, x0 + 2 * box_w + 3 * pad, y0 + 2 * box_h + int(58 * scale)),
            radius=int(12 * scale),
            fill=(8, 12, 20, 170),
        )
        draw.text((x0 + pad, y0 + int(6 * scale)), "foot contact", font=f_body, fill=_GREY)
        for k, leg in enumerate(LEG_NAMES):
            r, c = divmod(k, 2)
            bx = x0 + pad + c * (box_w + pad)
            by = y0 + int(40 * scale) + r * (box_h + int(6 * scale))
            on = bool(contacts[k])
            draw.rounded_rectangle(
                (bx, by, bx + box_w, by + box_h),
                radius=int(8 * scale),
                fill=(*_CYAN, 230) if on else (50, 58, 68, 230),
            )
            draw.text((bx + int(14 * scale), by + int(4 * scale)), leg, font=f_body, fill=(10, 14, 20) if on else _GREY)
    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"))


def title_card(
    width: int,
    height: int,
    heading: str,
    lines: Sequence[str],
    background: np.ndarray | None = None,
) -> np.ndarray:
    """A full-frame card; if a background frame is given it is darkened underneath."""
    if background is not None:
        base = Image.fromarray((background.astype(np.float32) * 0.35).astype(np.uint8))
    else:
        base = Image.new("RGB", (width, height), (10, 14, 22))
    draw = ImageDraw.Draw(base)
    scale = height / 720
    f_head, f_body = _font(int(58 * scale)), _font(int(28 * scale))
    y = int(height * 0.30)
    draw.text((width // 2, y), heading, font=f_head, fill=_ORANGE, anchor="mm")
    y += int(80 * scale)
    for line in lines:
        draw.text((width // 2, y), line, font=f_body, fill=_WHITE, anchor="mm")
        y += int(46 * scale)
    return np.asarray(base)


def image_card(image: np.ndarray, width: int, height: int, caption: str) -> np.ndarray:
    """Letterbox an arbitrary image (e.g. a training curve) into a video frame."""
    canvas = Image.new("RGB", (width, height), (10, 14, 22))
    img = Image.fromarray(image).convert("RGB")
    scale = height / 720
    max_w, max_h = int(width * 0.9), int(height * 0.8)
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    canvas.paste(img, ((width - img.width) // 2, int(80 * scale)))
    ImageDraw.Draw(canvas).text(
        (width // 2, int(42 * scale)), caption, font=_font(int(32 * scale)), fill=_ORANGE, anchor="mm"
    )
    return np.asarray(canvas)


def _hud_lines(cmd, vx_avg: float, yaw_rate: float, height: float, x_pos: float, t: float, slow: int) -> list[str]:
    return [
        f"command   vx {cmd[0]:+.2f} m/s   yaw {cmd[2]:+.2f} rad/s",
        f"measured  vx {vx_avg:+.2f} m/s   yaw {yaw_rate:+.2f} rad/s",
        f"height {height:.3f} m   distance {x_pos:5.2f} m",
        f"sim time {t:5.2f} s" + (f"   ({slow}x slow motion)" if slow > 1 else ""),
    ]


def _substep_telemetry(env) -> tuple[np.ndarray, float, float, float]:
    """Foot contacts, base height, yaw rate and x position at the current physics step."""
    _, _, ang_vel_b, _, height = env._base_state()
    return env._contact_state()[0].copy(), height, float(ang_vel_b[2]), float(env.data.qpos[0])


def record_rollout(
    policy: Callable[[np.ndarray], np.ndarray] | None,
    env_cfg: QuadBotEnvConfig,
    command_schedule: Sequence[tuple[float, tuple[float, float, float]]],
    title: str,
    width: int = 1280,
    height: int = 720,
    seed: int = 7,
    camera: dict | None = None,
    slow_motion: bool = False,
    warmup_seconds: float = 0.0,
    sink: Callable[[np.ndarray], None] | None = None,
    hud: bool = True,
) -> tuple[list[np.ndarray], dict]:
    """Roll out ``policy`` (uniform random actions if None) and render HUD-annotated frames.

    ``command_schedule`` is a list of (duration_seconds, (vx, vy, yaw_rate)).
    ``slow_motion`` renders every physics step instead of every control step, so playback is
    ``frame_skip`` (5x) slower while still showing the real intermediate motion.
    ``warmup_seconds`` of unrendered simulation (first command) run before recording starts.
    ``sink`` receives each frame (e.g. an :class:`Mp4Writer`); frames are then streamed instead
    of collected and the returned list stays empty, so memory use is constant.
    """
    env = gym.make(ENV_ID, config=env_cfg, render_mode="rgb_array")
    base = env.unwrapped
    for key, value in (camera or {}).items():
        setattr(base.cfg, f"camera_{key}", value)
    frames: list[np.ndarray] = []
    emit = frames.append if sink is None else sink
    env.action_space.seed(seed)
    act = (lambda _obs: env.action_space.sample()) if policy is None else policy

    first_cmd = command_schedule[0][1]
    obs, _ = env.reset(seed=seed, options={"command": first_cmd})
    for _ in range(int(round(warmup_seconds / base.dt))):
        obs, _, term, trunc, _ = env.step(act(obs))
        if term or trunc:
            obs, _ = env.reset(options={"command": first_cmd})

    slow = base.frame_skip if slow_motion else 1
    substeps: list[tuple] = []
    if slow_motion:
        base.substep_callback = lambda e: substeps.append((e.render_frame(width, height), *_substep_telemetry(e)))
    sim_dt = base.model.opt.timestep
    vx_hist: list[float] = []
    falls, n_frames, t = 0, 0, 0.0
    try:
        for duration, cmd in command_schedule:
            base.set_command(*cmd)
            for _ in range(int(round(duration / base.dt))):
                obs, _, term, trunc, info = env.step(act(obs))
                vx_hist.append(info["x_velocity"])
                vx_avg = float(np.mean(vx_hist[-25:]))
                if slow_motion:
                    shots = [
                        (frame, contacts, _hud_lines(cmd, vx_avg, yaw, h, x, t + (k + 1) * sim_dt, slow))
                        for k, (frame, contacts, h, yaw, x) in enumerate(substeps)
                    ]
                    substeps.clear()
                else:
                    lines = _hud_lines(
                        cmd, vx_avg, info["yaw_rate"], info["base_height"], info["x_position"], t + base.dt, 1
                    )
                    shots = [(base.render_frame(width, height), base.foot_contacts, lines)]
                for frame, contacts, lines in shots:
                    emit(draw_hud(frame, title, lines, contacts) if hud else frame)
                n_frames += len(shots)
                t += base.dt
                if term or trunc:
                    falls += int(term)
                    obs, _ = env.reset(options={"command": cmd})
        stats = dict(
            mean_vx=float(np.mean(vx_hist)), falls=falls, distance=float(base.data.qpos[0]), frames=n_frames
        )
    finally:
        base.substep_callback = None
        env.close()
    return frames, stats


class Mp4Writer:
    """Streaming H.264/yuv420p MP4 writer (plays on phones, browsers and Colab).

    Use as a context manager and call it with RGB frames; memory use does not grow with length.
    ``quality`` follows imageio (0-10); 6 corresponds to x264 CRF 20.
    """

    def __init__(self, path: str, fps: int = 50, quality: int = 6) -> None:
        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path, self.fps, self.frames = path, fps, 0
        self._writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            quality=quality,
            pixelformat="yuv420p",
            macro_block_size=1,
            ffmpeg_params=["-movflags", "+faststart"],
        )

    def __call__(self, frame: np.ndarray) -> None:
        self._writer.append_data(frame)
        self.frames += 1

    @property
    def seconds(self) -> float:
        return self.frames / self.fps

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "Mp4Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_mp4(frames: Sequence[np.ndarray], path: str, fps: int = 50) -> str:
    """Encode a list of frames (see :class:`Mp4Writer` for streaming)."""
    with Mp4Writer(path, fps) as writer:
        for frame in frames:
            writer(frame)
    return path


def load_training_curves(run_dir: str) -> dict[str, np.ndarray]:
    """Collect curves written during training (metrics history + EvalCallback results)."""
    import json

    out: dict[str, np.ndarray] = {}
    hist_path = os.path.join(run_dir, "training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            hist = json.load(f)
        for key in ("timesteps", "ep_rew", "ep_len", "vx", "vx_err"):
            out[key] = np.array([h[key] for h in hist], dtype=float)
    eval_path = os.path.join(run_dir, "eval", "evaluations.npz")
    if os.path.exists(eval_path):
        with np.load(eval_path) as z:
            out["eval_timesteps"] = z["timesteps"].astype(float)
            out["eval_return"] = z["results"].mean(axis=1)
    return out


def training_curve_figure(run_dir: str, dpi: int = 110):
    """Matplotlib figure: episode return and velocity-tracking error vs. timesteps."""
    import matplotlib.pyplot as plt

    c = load_training_curves(run_dir)
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=dpi)
        fig.patch.set_facecolor("#0a0e16")
        ax = axes[0]
        if "timesteps" in c:
            ax.plot(c["timesteps"] / 1e6, c["ep_rew"], color="#f5851f", lw=2,
                    label="training (mean of last 100 episodes)")
        if "eval_timesteps" in c:
            ax.plot(c["eval_timesteps"] / 1e6, c["eval_return"], "o", color="#1ad9ff", ms=4,
                    label="evaluation (deterministic)")
        ax.set(title="Episode return", xlabel="environment steps [millions]", ylabel="return")
        ax.legend(loc="lower right", fontsize=9)
        ax = axes[1]
        if "timesteps" in c:
            ax.plot(c["timesteps"] / 1e6, c["vx_err"], color="#f5851f", lw=2,
                    label="|commanded vx - actual vx| [m/s]")
            ax.plot(c["timesteps"] / 1e6, c["ep_len"] / 1000.0, color="#9aa4ae", lw=1.5, ls="--",
                    label="episode length / max length")
        ax.set(title="Command tracking & survival", xlabel="environment steps [millions]")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="center right", fontsize=9)
        for ax in axes:
            ax.set_facecolor("#111827")
            ax.grid(alpha=0.2)
        fig.tight_layout()
    return fig


def figure_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def footfall_figure(
    contacts: np.ndarray,
    dt: float,
    title: str = "Footfall pattern",
    figsize: tuple[float, float] = (12, 3.2),
    dpi: int = 110,
):
    """Gait diagram: one bar per stance phase of each foot; diagonal pairs share a colour."""
    import matplotlib.pyplot as plt

    contacts = np.asarray(contacts, dtype=bool).reshape(-1, len(LEG_NAMES))
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor("#0a0e16")
        ax.set_facecolor("#111827")
        for k, leg in enumerate(LEG_NAMES):
            edges = np.flatnonzero(np.diff(np.r_[0, contacts[:, k].astype(np.int8), 0]))
            spans = [(a * dt, (b - a) * dt) for a, b in zip(edges[::2], edges[1::2])]
            color = "#1ad9ff" if leg in ("FL", "RR") else "#f5851f"
            ax.broken_barh(spans, (k - 0.35, 0.7), facecolors=color)
        ax.set_yticks(range(len(LEG_NAMES)))
        ax.set_yticklabels(LEG_NAMES)
        ax.set_ylim(len(LEG_NAMES) - 0.5, -0.5)
        ax.set_xlim(0, max(len(contacts), 1) * dt)
        ax.set(xlabel="time [s]  (bars = foot on the ground)", title=title)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
    return fig


def make_showcase_video(
    policy: Callable[[np.ndarray], np.ndarray],
    env_cfg: QuadBotEnvConfig,
    out_path: str,
    subtitle_lines: Sequence[str] = (),
    curve_image: np.ndarray | None = None,
    extra_cards: Sequence[tuple[np.ndarray, str]] = (),
    results_lines: Sequence[str] = (),
    gait_name: str | None = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 50,
    include_turning: bool = True,
    seed: int = 7,
) -> dict:
    """Compose the demo video and stream it straight into ``out_path`` (constant memory).

    Title card -> untrained robot (random actions) -> trained policy on forward-speed commands
    -> yaw-rate commands -> slow-motion gait close-up (every physics step rendered) -> learning
    curve -> ``extra_cards`` [(image, caption)] -> results card (``results_lines``).
    """
    stats: dict[str, dict] = {}
    common = dict(width=width, height=height, seed=seed)
    chase = dict(distance=1.7, azimuth=140.0, elevation=-14.0)
    backdrop, _ = record_rollout(
        policy, env_cfg, [(env_cfg.control_dt, (0.7, 0.0, 0.0))], "",
        camera=chase, warmup_seconds=2.0, hud=False, **common,
    )
    gait_title = f"{gait_name} gait".upper() if gait_name else "GAIT"

    with Mp4Writer(out_path, fps) as video:

        def hold(frame: np.ndarray, seconds: float) -> None:
            for _ in range(int(round(seconds * fps))):
                video(frame)

        hold(title_card(
            width, height, "QuadBot",
            ["Custom 12-DoF quadruped learns to walk with reinforcement learning", *subtitle_lines],
            background=backdrop[-1],
        ), 3.0)
        _, stats["random"] = record_rollout(
            None, env_cfg, [(3.0, (0.7, 0.0, 0.0))], "BEFORE TRAINING - random actions",
            camera=chase, sink=video, **common,
        )
        _, stats["speed_sweep"] = record_rollout(
            policy, env_cfg, [(3.0, (0.4, 0.0, 0.0)), (3.0, (0.7, 0.0, 0.0)), (4.0, (1.0, 0.0, 0.0))],
            "AFTER TRAINING - speed commands", camera=chase, sink=video, **common,
        )
        if include_turning:
            _, stats["turning"] = record_rollout(
                policy, env_cfg, [(4.0, (0.6, 0.0, 0.5)), (4.0, (0.6, 0.0, -0.5))],
                "AFTER TRAINING - yaw-rate commands",
                camera=dict(distance=2.6, azimuth=110.0, elevation=-38.0), sink=video, **common,
            )
        _, stats["slow_motion"] = record_rollout(
            policy, env_cfg, [(1.6, (0.8, 0.0, 0.0))], f"{gait_title} - {env_cfg.frame_skip}x slow motion",
            camera=dict(distance=1.15, azimuth=90.0, elevation=-8.0),
            slow_motion=True, warmup_seconds=1.0, sink=video, **common,
        )
        if curve_image is not None:
            hold(image_card(curve_image, width, height, "Training progress"), 4.0)
        for image, caption in extra_cards:
            hold(image_card(image, width, height, caption), 4.0)
        if results_lines:
            hold(title_card(width, height, "Results", list(results_lines)), 5.0)
    stats["video"] = dict(path=out_path, seconds=video.seconds, frames=video.frames)
    return stats
