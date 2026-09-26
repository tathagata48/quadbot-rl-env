"""Dependency-light policy inference (NumPy only).

A trained Stable-Baselines3 PPO actor is exported to a single ``.npz`` holding
the MLP weights and the observation-normalisation statistics. Loading it needs
neither PyTorch nor pickle, so the file works across SB3/NumPy versions and is
a convenient starting point for on-robot deployment.
"""
from __future__ import annotations

import os

import numpy as np


class NumpyPolicy:
    """Deterministic PPO actor: raw observation -> action in [-1, 1]."""

    def __init__(self, path: str | os.PathLike) -> None:
        with np.load(path, allow_pickle=False) as z:
            n = int(z["n_layers"])
            self.layers = [(z[f"W{i}"].astype(np.float64), z[f"b{i}"].astype(np.float64)) for i in range(n)]
            self.obs_mean = z["obs_mean"].astype(np.float64)
            self.obs_std = np.sqrt(z["obs_var"].astype(np.float64) + float(z["epsilon"]))
            self.clip_obs = float(z["clip_obs"])
            self.activation = str(z["activation"])
        if self.activation not in ("elu", "tanh", "relu"):
            raise ValueError(f"Unsupported activation {self.activation!r}")

    @property
    def obs_dim(self) -> int:
        return self.layers[0][0].shape[1]

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "elu":
            return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))
        if self.activation == "tanh":
            return np.tanh(x)
        return np.maximum(x, 0.0)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.clip((np.asarray(obs, dtype=np.float64) - self.obs_mean) / self.obs_std, -self.clip_obs, self.clip_obs)
        for w, b in self.layers[:-1]:
            x = self._act(x @ w.T + b)
        w, b = self.layers[-1]
        return np.clip(x @ w.T + b, -1.0, 1.0).astype(np.float32)
