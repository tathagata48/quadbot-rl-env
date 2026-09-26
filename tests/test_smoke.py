"""Smoke tests: the environment builds and steps, and the exported policy walks.

These run on CPU with no rendering, so they work on a bare CI runner.
"""
from __future__ import annotations

import pathlib

import gymnasium as gym
import numpy as np
import pytest

from quadbot import ENV_ID, QuadBotEnv, QuadBotEnvConfig
from quadbot.policy import NumpyPolicy

POLICY_PATH = pathlib.Path(__file__).resolve().parents[1] / "runs" / "main" / "quadbot_policy.npz"


def test_env_is_registered():
    assert ENV_ID in gym.registry


def test_spaces_match_the_documented_mdp():
    env = QuadBotEnv()
    try:
        assert env.action_space.shape == (12,)  # 3 joints x 4 legs
        assert env.observation_space.shape == (48,)
    finally:
        env.close()


def test_reset_and_step_are_finite():
    env = gym.make(ENV_ID)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (48,)
        assert np.all(np.isfinite(obs))

        for _ in range(50):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)
            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()


def test_command_can_be_pinned_through_reset_options():
    env = QuadBotEnv(config=QuadBotEnvConfig(command_yaw_range=(-0.6, 0.6)))
    try:
        env.reset(seed=0, options={"command": (0.7, 0.0, 0.3)})
        assert np.allclose(env.command, (0.7, 0.0, 0.3))
    finally:
        env.close()


@pytest.mark.skipif(not POLICY_PATH.exists(), reason="reference policy not available")
def test_pretrained_policy_tracks_its_command_without_falling():
    """The shipped policy should hold ~0.7 m/s for a full 20 s episode."""
    policy = NumpyPolicy(POLICY_PATH)
    assert policy.obs_dim == 48

    env = gym.make(ENV_ID)
    try:
        obs, _ = env.reset(seed=0, options={"command": (0.7, 0.0, 0.0)})
        speeds, fell = [], False
        for _ in range(1000):
            obs, _reward, terminated, truncated, _info = env.step(policy(obs))
            speeds.append(env.unwrapped.data.qvel[0])
            if terminated:
                fell = True
                break
            if truncated:
                break
    finally:
        env.close()

    assert not fell, "policy fell over while tracking 0.7 m/s"
    assert float(np.mean(speeds)) == pytest.approx(0.7, abs=0.15)
