"""PPO training utilities for QuadBot (Stable-Baselines3)."""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from . import ENV_ID
from .env import QuadBotEnvConfig


@dataclass
class TrainConfig:
    """PPO hyper-parameters tuned for CPU training (1-4 cores)."""

    total_timesteps: int = 6_000_000
    n_envs: int = 16
    seed: int = 42
    n_steps: int = 256  # per env -> 4096 samples per update
    batch_size: int = 1024
    n_epochs: int = 5
    learning_rate: float = 3e-4
    final_learning_rate: float = 5e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.002
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    target_kl: float | None = 0.03
    net_arch: tuple[int, ...] = (256, 256, 128)
    log_std_init: float = -0.5
    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0
    use_subproc: bool | None = None  # None -> SubprocVecEnv only with >= 4 CPU cores
    eval_every_steps: int = 250_000
    checkpoint_every_steps: int = 500_000
    run_dir: str = "runs/quadbot_ppo"
    env: QuadBotEnvConfig = field(default_factory=QuadBotEnvConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def _env_factory(rank: int, seed: int, env_cfg: QuadBotEnvConfig, monitor_dir: str | None) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        import quadbot  # noqa: F401  (registers QuadBot-v0 inside worker processes)

        env = gym.make(ENV_ID, config=env_cfg)
        env = Monitor(
            env,
            filename=None if monitor_dir is None else os.path.join(monitor_dir, str(rank)),
            info_keywords=("x_position",),
        )
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


def make_vec_env(
    n_envs: int,
    seed: int,
    env_cfg: QuadBotEnvConfig,
    monitor_dir: str | None = None,
    use_subproc: bool | None = None,
) -> VecEnv:
    if monitor_dir is not None:
        os.makedirs(monitor_dir, exist_ok=True)
    fns = [_env_factory(i, seed, env_cfg, monitor_dir) for i in range(n_envs)]
    if use_subproc is None:
        use_subproc = (os.cpu_count() or 1) >= 4 and n_envs > 1
    return SubprocVecEnv(fns) if use_subproc else DummyVecEnv(fns)


def _lr_schedule(start: float, end: float) -> Callable[[float], float]:
    # SB3 passes progress_remaining: 1.0 at the start, 0.0 at the end.
    return lambda progress: end + (start - end) * progress


class LocomotionMetricsCallback(BaseCallback):
    """Logs forward speed, command-tracking error and throughput to TensorBoard/stdout.

    ``history`` (one entry per PPO update) is also written to ``history_path`` at every
    printout, so the learning curve survives an interrupted run.
    """

    def __init__(
        self,
        print_every_steps: int = 100_000,
        verbose: int = 1,
        history_path: str | os.PathLike | None = None,
        history: list[dict[str, float]] | None = None,
    ) -> None:
        super().__init__(verbose)
        self.print_every_steps = print_every_steps
        self.history_path = None if history_path is None else Path(history_path)
        self.history: list[dict[str, float]] = list(history or [])
        self._vx: list[float] = []
        self._err: list[float] = []
        self._t0 = time.time()
        self._start_steps = 0
        self._next_print = print_every_steps

    def _on_training_start(self) -> None:
        self._t0 = time.time()
        self._start_steps = self.num_timesteps
        self._next_print = (self.num_timesteps // self.print_every_steps + 1) * self.print_every_steps

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self._vx.append(info["x_velocity"])
            self._err.append(abs(info["command_vx"] - info["x_velocity"]))
        return True

    def _on_rollout_end(self) -> None:
        if not self._vx:
            return
        vx, err = float(np.mean(self._vx)), float(np.mean(self._err))
        elapsed = max(time.time() - self._t0, 1e-6)
        fps = (self.num_timesteps - self._start_steps) / elapsed
        self.logger.record("locomotion/forward_velocity", vx)
        self.logger.record("locomotion/vx_tracking_abs_error", err)
        self.logger.record("locomotion/steps_per_second", fps)
        ep_buf = self.model.ep_info_buffer
        ep_rew = float(np.mean([e["r"] for e in ep_buf])) if ep_buf else float("nan")
        ep_len = float(np.mean([e["l"] for e in ep_buf])) if ep_buf else float("nan")
        self.history.append(
            dict(timesteps=self.num_timesteps, ep_rew=ep_rew, ep_len=ep_len, vx=vx, vx_err=err)
        )
        if self.num_timesteps >= self._next_print:
            self._next_print = (self.num_timesteps // self.print_every_steps + 1) * self.print_every_steps
            self.save_history()
            if self.verbose:
                print(
                    f"[{self.num_timesteps:>9,d} steps | {elapsed / 60:5.1f} min | {fps:5.0f} sps] "
                    f"ep_return={ep_rew:8.1f} ep_len={ep_len:6.1f} vx={vx:5.2f} m/s |vx err|={err:4.2f}",
                    flush=True,
                )
        self._vx.clear()
        self._err.clear()

    def save_history(self) -> None:
        if self.history_path is not None:
            self.history_path.write_text(json.dumps(self.history))


def build_model(cfg: TrainConfig, venv: VecEnv, tensorboard_log: str | None) -> PPO:
    policy_kwargs = dict(
        net_arch=dict(pi=list(cfg.net_arch), vf=list(cfg.net_arch)),
        activation_fn=torch.nn.ELU,
        log_std_init=cfg.log_std_init,
    )
    return PPO(
        "MlpPolicy",
        venv,
        learning_rate=_lr_schedule(cfg.learning_rate, cfg.final_learning_rate),
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        target_kl=cfg.target_kl,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=cfg.seed,
        device="cpu",  # small MLPs train faster on CPU than on a GPU
        verbose=0,
    )


class SaveNormalizerOnBestCallback(BaseCallback):
    """``EvalCallback(callback_on_new_best=...)`` hook: saves the observation normaliser at the
    moment a new best model is written, so ``best/`` always holds a matching pair."""

    def __init__(self, path: str | os.PathLike) -> None:
        super().__init__(verbose=0)
        self.path = str(path)

    def _on_step(self) -> bool:
        normalizer = self.model.get_vec_normalize_env()
        if normalizer is not None:
            normalizer.save(self.path)
        return True


def latest_checkpoint(run_dir: str | os.PathLike) -> tuple[Path, Path, int] | None:
    """Newest (model .zip, normaliser .pkl, timesteps) written by the checkpoint callback, if any."""
    found = []
    for zip_path in Path(run_dir, "checkpoints").glob("quadbot_*_steps.zip"):
        steps = int(zip_path.stem.split("_")[1])
        pkl = zip_path.with_name(f"quadbot_vecnormalize_{steps}_steps.pkl")
        if pkl.exists():
            found.append((steps, zip_path, pkl))
    if not found:
        return None
    steps, zip_path, pkl = max(found)
    return zip_path, pkl, steps


def train(
    cfg: TrainConfig, progress_bar: bool = False, resume: bool = False
) -> tuple[PPO, VecNormalize, LocomotionMetricsCallback]:
    """Train PPO and write model, normaliser stats and logs into ``cfg.run_dir``.

    With ``resume=True`` training continues from the newest checkpoint in ``run_dir`` (if any),
    keeping the learning-rate schedule, the learning curve and the evaluation log.
    """
    run = Path(cfg.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(cfg.to_json())
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    ckpt = latest_checkpoint(run) if resume else None
    start = ckpt[2] if ckpt else 0

    monitor_dir = run / "monitor" / f"resumed_at_{start}" if start else run / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    venv = make_vec_env(cfg.n_envs, cfg.seed + start, cfg.env, str(monitor_dir), cfg.use_subproc)
    if ckpt:
        venv = VecNormalize.load(str(ckpt[1]), venv)
        model = PPO.load(str(ckpt[0]), env=venv, device="cpu")
        print(f"Resuming from {ckpt[0].name} ({start:,} of {cfg.total_timesteps:,} steps)", flush=True)
    else:
        venv = VecNormalize(
            venv, norm_obs=cfg.norm_obs, norm_reward=cfg.norm_reward, clip_obs=cfg.clip_obs, gamma=cfg.gamma
        )
        model = build_model(cfg, venv, str(run / "tb"))
    eval_env = VecNormalize(
        make_vec_env(1, cfg.seed + 10_000, cfg.env, None, use_subproc=False),
        norm_obs=cfg.norm_obs,
        norm_reward=False,
        clip_obs=cfg.clip_obs,
        training=False,
    )

    hist_path = run / "training_history.json"
    history = []
    if start and hist_path.exists():
        history = [h for h in json.loads(hist_path.read_text()) if h["timesteps"] <= start]
    metrics = LocomotionMetricsCallback(history_path=hist_path, history=history)
    callbacks = [metrics]
    if cfg.eval_every_steps:
        eval_cb = EvalCallback(
            eval_env,
            n_eval_episodes=3,
            eval_freq=max(cfg.eval_every_steps // cfg.n_envs, 1),
            best_model_save_path=str(run / "best"),
            callback_on_new_best=SaveNormalizerOnBestCallback(run / "best" / "vecnormalize.pkl"),
            log_path=str(run / "eval"),
            deterministic=True,
            verbose=0,
        )
        old_eval = run / "eval" / "evaluations.npz"
        if start and old_eval.exists():  # keep the evaluation curve up to the checkpoint
            with np.load(old_eval) as z:
                keep = z["timesteps"] <= start
                eval_cb.evaluations_timesteps = z["timesteps"][keep].tolist()
                eval_cb.evaluations_results = z["results"][keep].tolist()
                eval_cb.evaluations_length = z["ep_lengths"][keep].tolist()
        callbacks.append(eval_cb)
    if cfg.checkpoint_every_steps:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(cfg.checkpoint_every_steps // cfg.n_envs, 1),
                save_path=str(run / "checkpoints"),
                name_prefix="quadbot",
                save_vecnormalize=True,
            )
        )

    t0 = time.time()
    remaining = cfg.total_timesteps - model.num_timesteps
    if remaining > 0:
        model.learn(
            total_timesteps=remaining,
            callback=CallbackList(callbacks),
            progress_bar=progress_bar,
            reset_num_timesteps=ckpt is None,
        )
    minutes = (time.time() - t0) / 60
    model.save(run / "final_model")
    venv.save(str(run / "vecnormalize.pkl"))
    metrics.save_history()
    print(f"Training finished in {minutes:.1f} min ({model.num_timesteps:,} steps) -> {run}", flush=True)
    eval_env.close()
    return model, venv, metrics


def export_policy_npz(model: PPO, normalizer: VecNormalize, path: str | os.PathLike) -> str:
    """Export the deterministic actor + observation normaliser to a portable .npz."""
    policy = model.policy
    act_fn = policy.activation_fn
    act_name = {torch.nn.ELU: "elu", torch.nn.Tanh: "tanh", torch.nn.ReLU: "relu"}[act_fn]
    linears = [m for m in policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    linears.append(policy.action_net)
    arrays: dict[str, np.ndarray] = {}
    for i, layer in enumerate(linears):
        arrays[f"W{i}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    arrays.update(
        n_layers=np.array(len(linears)),
        activation=np.array(act_name),
        obs_mean=normalizer.obs_rms.mean.astype(np.float32),
        obs_var=normalizer.obs_rms.var.astype(np.float32),
        epsilon=np.array(normalizer.epsilon, dtype=np.float32),
        clip_obs=np.array(normalizer.clip_obs, dtype=np.float32),
    )
    path = str(path)
    np.savez(path, **arrays)
    return path if path.endswith(".npz") else path + ".npz"


class QuadBotPolicy:
    """Inference wrapper: raw observation in, action out (applies saved obs normalisation)."""

    def __init__(self, model_path: str | os.PathLike, vecnormalize_path: str | os.PathLike) -> None:
        custom_objects = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}
        self.model = PPO.load(str(model_path), device="cpu", custom_objects=custom_objects)
        with open(vecnormalize_path, "rb") as f:
            self.normalizer: VecNormalize = pickle.load(f)
        self.normalizer.training = False
        self.normalizer.norm_reward = False

    def __call__(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        norm_obs = self.normalizer.normalize_obs(np.asarray(obs, dtype=np.float32))
        action, _ = self.model.predict(norm_obs, deterministic=deterministic)
        return action


def evaluate_policy_metrics(
    policy: Callable[[np.ndarray], np.ndarray],
    env_cfg: QuadBotEnvConfig,
    commands: tuple[float, ...] = (0.4, 0.7, 1.0),
    episodes_per_command: int = 2,
    seed: int = 123,
) -> list[dict[str, Any]]:
    """Run full episodes at fixed forward-speed commands and report tracking quality."""
    results = []
    env = gym.make(ENV_ID, config=env_cfg)
    for i, vx_cmd in enumerate(commands):
        for ep in range(episodes_per_command):
            obs, _ = env.reset(seed=seed + 100 * i + ep, options={"command": (vx_cmd, 0.0, 0.0)})
            ret, n, vxs, done = 0.0, 0, [], False
            while not done:
                obs, r, term, trunc, info = env.step(policy(obs))
                ret += r
                n += 1
                vxs.append(info["x_velocity"])
                done = term or trunc
            results.append(
                dict(
                    command_vx=vx_cmd,
                    episode=ep,
                    return_=round(ret, 1),
                    steps=n,
                    fell=bool(term),
                    mean_vx=round(float(np.mean(vxs[50:] or vxs)), 3),
                    distance_m=round(info["x_position"], 2),
                )
            )
    env.close()
    return results


def evaluation_summary_lines(results: list[dict[str, Any]], gait: dict[str, Any] | None = None) -> list[str]:
    """Short human-readable summary of :func:`evaluate_policy_metrics` (and :func:`gait_analysis`)."""
    falls = sum(bool(r["fell"]) for r in results)
    lines = [f"{len(results)} evaluation episodes, deterministic policy: {falls} fall{'' if falls == 1 else 's'}"]
    for vx in sorted({r["command_vx"] for r in results}):
        rows = [r for r in results if r["command_vx"] == vx]
        measured = float(np.mean([r["mean_vx"] for r in rows]))
        distance = float(np.mean([r["distance_m"] for r in rows]))
        lines.append(f"command {vx:.2f} m/s  ->  measured {measured:.2f} m/s   ({distance:.1f} m per episode)")
    if gait is not None:
        lines.append(
            f"gait: {gait['label']}  -  diagonal feet in phase {gait['diagonal_sync']:.0%} of the time, "
            f"{gait['stride_hz']:.1f} strides/s"
        )
    return lines


def gait_analysis(
    policy: Callable[[np.ndarray], np.ndarray],
    env_cfg: QuadBotEnvConfig,
    command: tuple[float, float, float] = (0.7, 0.0, 0.0),
    seconds: float = 8.0,
    settle_seconds: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Foot-contact statistics of a deterministic rollout and the gait they imply.

    ``*_sync`` is the fraction of time two feet are in the same phase (both in stance or both
    in swing); ``*_corr`` is the correlation of their contact signals, which does not depend on
    the duty factor and decides the label: diagonal pairs -> trot, same-side pairs -> pace,
    front/rear pairs -> bound. ``contacts`` is a (T, 4) bool array in FL, FR, RL, RR order.
    """
    env = gym.make(ENV_ID, config=env_cfg)
    base = env.unwrapped
    dt = base.dt
    obs, _ = env.reset(seed=seed, options={"command": tuple(command)})
    settle = int(round(settle_seconds / dt))
    rows, fell = [], False
    for i in range(settle + int(round(seconds / dt))):
        obs, _, term, trunc, _ = env.step(policy(obs))
        if i >= settle:
            rows.append(base.foot_contacts.copy())
        if term or trunc:
            fell = bool(term)
            break
    env.close()

    c = np.array(rows, dtype=bool).reshape(-1, 4)

    def pair(a: int, b: int) -> tuple[float, float]:
        if len(c) == 0:
            return 0.0, 0.0
        x, y = c[:, a].astype(float), c[:, b].astype(float)
        corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0
        return float(np.mean(x == y)), corr

    stats = {name: [pair(*ij) for ij in pairs] for name, pairs in (
        ("diagonal", ((0, 3), (1, 2))), ("lateral", ((0, 2), (1, 3))), ("fore_hind", ((0, 1), (2, 3))),
    )}
    sync = {name: float(np.mean([s for s, _ in v])) for name, v in stats.items()}
    corr = {name: float(np.mean([r for _, r in v])) for name, v in stats.items()}
    duty = c.mean(axis=0) if len(c) else np.zeros(4)
    touchdowns = np.count_nonzero(np.diff(c.astype(np.int8), axis=0) == 1, axis=0) if len(c) > 1 else np.zeros(4)
    duration = len(c) * dt

    names = {"diagonal": "trot", "lateral": "pace", "fore_hind": "bound"}
    best = max(corr, key=corr.get)
    if touchdowns.sum() == 0:
        label = "fell" if fell else "standing"
    elif corr[best] >= 0.5:
        label = names[best]
    else:
        label = "walk" if duty.mean() >= 0.5 else "irregular"
    return dict(
        label=label, command=tuple(float(v) for v in command), seconds=duration, dt=dt, fell=fell,
        diagonal_sync=sync["diagonal"], lateral_sync=sync["lateral"], fore_hind_sync=sync["fore_hind"],
        diagonal_corr=corr["diagonal"], lateral_corr=corr["lateral"], fore_hind_corr=corr["fore_hind"],
        duty_factor=[float(x) for x in duty],
        stride_hz=float(touchdowns.mean() / duration) if duration > 0 else 0.0,
        contacts=c,
    )
