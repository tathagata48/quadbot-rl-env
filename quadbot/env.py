"""Gymnasium environment: QuadBot learns to walk by tracking velocity commands.

MDP summary
-----------
* Action (12,): joint-position offsets in [-1, 1], scaled by ``action_scale``
  and added to the nominal standing pose. Low-level PD servos run at 250 Hz,
  the policy at 50 Hz.
* Observation (48,): projected gravity (3), base linear velocity (3), base
  angular velocity (3), command (3), joint positions relative to the nominal
  pose (12), joint velocities (12), previous action (12). All in the body frame.
* Reward: legged_gym-style sum of a velocity-tracking term and regularisers
  (see ``DEFAULT_REWARD_SCALES``), optionally clipped at zero.
* Termination: torso touches the ground, torso too low, or tilt > 60 deg.
* Truncation: handled by the ``TimeLimit`` wrapper added at registration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle

from .robot import FOOT_GEOM_NAMES, JOINT_NAMES, LEG_NAMES, RobotSpec, build_quadbot_xml

DEFAULT_REWARD_SCALES: dict[str, float] = {
    "tracking_lin_vel": 1.5,
    "tracking_ang_vel": 0.5,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "orientation": -5.0,
    "base_height": -30.0,
    "torques": -1e-4,
    "dof_acc": -2.5e-7,
    "action_rate": -0.01,
    "feet_air_time": 1.0,
    "feet_slip": -0.1,
    "collision": -1.0,
    "dof_pos_limits": -10.0,
    "abad_deviation": -0.5,
}


@dataclass
class QuadBotEnvConfig:
    """Everything that defines the task. Defaults reproduce the reference run."""

    robot: RobotSpec = field(default_factory=RobotSpec)
    # Simulation / control
    sim_timestep: float = 0.004  # 250 Hz physics
    frame_skip: int = 5  # 50 Hz policy
    action_scale: float = 0.3  # rad per unit action
    # Commands: (vx [m/s], vy [m/s], yaw rate [rad/s]) sampled uniformly
    command_vx_range: tuple[float, float] = (0.3, 1.0)
    command_vy_range: tuple[float, float] = (0.0, 0.0)
    command_yaw_range: tuple[float, float] = (0.0, 0.0)
    command_resample_seconds: float = 10.0
    # Observations
    include_base_lin_vel: bool = True  # privileged in sim; replace with an estimator on hardware
    obs_noise_scale: float = 0.0  # 0 disables observation noise
    # Termination
    min_base_height: float = 0.16
    min_uprightness: float = 0.5  # cos(max tilt); 0.5 -> 60 degrees
    # Reset distribution
    reset_joint_noise: float = 0.1
    reset_base_vel_noise: float = 0.1
    # Domain randomisation (all off by default)
    domain_randomization: bool = False
    friction_range: tuple[float, float] = (0.5, 1.25)
    payload_range: tuple[float, float] = (-0.5, 1.5)  # kg added to torso
    motor_strength_range: tuple[float, float] = (0.9, 1.1)
    push_interval_seconds: float = 8.0
    push_max_velocity: float = 0.4
    # Reward
    reward_scales: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REWARD_SCALES))
    tracking_sigma: float = 0.25
    base_height_target: float = 0.30
    feet_air_time_target: float = 0.30
    soft_dof_pos_limit: float = 0.95  # fraction of joint range before penalty
    only_positive_rewards: bool = True
    # Rendering
    render_width: int = 640
    render_height: int = 480
    camera_distance: float = 1.6
    camera_azimuth: float = 140.0
    camera_elevation: float = -15.0

    @property
    def control_dt(self) -> float:
        return self.sim_timestep * self.frame_skip


# Observation scaling (matches common legged-robot practice).
_LIN_VEL_SCALE = 2.0
_ANG_VEL_SCALE = 0.25
_DOF_VEL_SCALE = 0.05
_COMMAND_SCALE = np.array([2.0, 2.0, 0.25])


class QuadBotEnv(gym.Env, EzPickle):
    """Velocity-command locomotion for the custom QuadBot quadruped."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        config: QuadBotEnvConfig | None = None,
        **config_overrides: Any,
    ) -> None:
        EzPickle.__init__(self, render_mode, config, **config_overrides)
        cfg = config or QuadBotEnvConfig()
        if config_overrides:
            cfg = replace(cfg, **config_overrides)
        self.cfg = cfg
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_string(
            build_quadbot_xml(cfg.robot, sim_timestep=cfg.sim_timestep)
        )
        self.data = mujoco.MjData(self.model)
        self.frame_skip = int(cfg.frame_skip)
        self.dt = cfg.control_dt
        self.metadata = {**self.metadata, "render_fps": int(round(1.0 / self.dt))}

        m = self.model
        self._torso_id = m.body("torso").id
        self._torso_geom = m.geom("torso").id
        self._floor_geom = m.geom("floor").id
        self._foot_geoms = np.array([m.geom(n).id for n in FOOT_GEOM_NAMES])
        self._non_foot_lut = np.ones(m.ngeom, dtype=bool)
        self._non_foot_lut[self._foot_geoms] = False
        self._foot_sites = np.array([m.site(f"{leg}_foot_site").id for leg in LEG_NAMES])
        self._touch_adr = np.array(
            [m.sensor_adr[m.sensor(f"{leg}_touch").id] for leg in LEG_NAMES]
        )
        qpos_adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINT_NAMES])
        qvel_adr = np.array([m.jnt_dofadr[m.joint(n).id] for n in JOINT_NAMES])
        # The generator emits joints contiguously; slices are faster than fancy indexing.
        assert np.all(np.diff(qpos_adr) == 1) and np.all(np.diff(qvel_adr) == 1)
        self._qs = slice(int(qpos_adr[0]), int(qpos_adr[-1]) + 1)
        self._vs = slice(int(qvel_adr[0]), int(qvel_adr[-1]) + 1)
        self._abad_idx = np.arange(0, 12, 3)

        self.default_joint_pos = np.tile(np.asarray(cfg.robot.default_joint_angles, float), 4)
        self._ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = m.actuator_ctrlrange[:, 1].copy()
        jnt_range = m.jnt_range[[m.joint(n).id for n in JOINT_NAMES]]
        mid, half = jnt_range.mean(axis=1), 0.5 * (jnt_range[:, 1] - jnt_range[:, 0])
        self._soft_lo = mid - cfg.soft_dof_pos_limit * half
        self._soft_hi = mid + cfg.soft_dof_pos_limit * half

        # Nominal values used by domain randomisation.
        self._nominal_torso_mass = float(m.body_mass[self._torso_id])
        self._nominal_gain = m.actuator_gainprm[:, 0].copy()
        self._nominal_bias = m.actuator_biasprm[:, :3].copy()
        self._friction_geoms = np.concatenate([[self._floor_geom], self._foot_geoms])
        self._nominal_friction = m.geom_friction[self._friction_geoms, 0].copy()

        self._max_steps_per_command = max(1, int(round(cfg.command_resample_seconds / self.dt)))
        self._push_every = (
            int(round(cfg.push_interval_seconds / self.dt)) if cfg.push_interval_seconds > 0 else 0
        )
        self._reward_terms = {k: v for k, v in cfg.reward_scales.items() if v != 0.0}

        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        obs_dim = 45 + (3 if cfg.include_base_lin_vel else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # Runtime state (initialised in reset).
        self.command = np.zeros(3)
        self._fixed_command: np.ndarray | None = None
        self._last_action = np.zeros(12)
        self._last_dof_vel = np.zeros(12)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4, dtype=bool)
        self._last_foot_pos = np.zeros((4, 3))
        self.foot_contacts = np.zeros(4, dtype=bool)
        self._step_count = 0
        self._episode_sums: dict[str, float] = {}
        self._renderer: mujoco.Renderer | None = None
        self._camera: mujoco.MjvCamera | None = None
        self._viewer = None
        # Optional hook f(env) called after every physics step (used for slow-motion video).
        self.substep_callback = None

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        m, d, rng, cfg = self.model, self.data, self.np_random, self.cfg
        mujoco.mj_resetData(m, d)

        if cfg.domain_randomization:
            self._randomize_dynamics()

        d.qpos[0:3] = (0.0, 0.0, cfg.robot.nominal_height + 0.01)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[self._qs] = self.default_joint_pos + rng.uniform(
            -cfg.reset_joint_noise, cfg.reset_joint_noise, 12
        )
        d.qvel[0:6] = rng.uniform(-cfg.reset_base_vel_noise, cfg.reset_base_vel_noise, 6)
        d.ctrl[:] = self.default_joint_pos
        mujoco.mj_forward(m, d)

        fixed = options.get("command")
        self._fixed_command = None if fixed is None else np.asarray(fixed, dtype=float).copy()
        self._resample_command()
        self._last_action[:] = 0.0
        self._last_dof_vel[:] = d.qvel[self._vs]
        self._feet_air_time[:] = 0.0
        self._last_contacts[:] = False
        self._last_foot_pos[:] = d.site_xpos[self._foot_sites]
        self._step_count = 0
        self._episode_sums = {k: 0.0 for k in self._reward_terms}

        obs = self._get_obs(self._base_state())
        if self.render_mode == "human":
            self.render()
        self.foot_contacts = np.zeros(4, dtype=bool)
        return obs, {"command_vx": float(self.command[0]), "command_yaw": float(self.command[2])}

    def step(self, action: np.ndarray):
        m, d, cfg = self.model, self.data, self.cfg
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        np.clip(
            self.default_joint_pos + cfg.action_scale * action, self._ctrl_lo, self._ctrl_hi, out=d.ctrl
        )
        if cfg.domain_randomization and self._push_every and self._step_count % self._push_every == 0 and self._step_count > 0:
            d.qvel[0:2] += self.np_random.uniform(-cfg.push_max_velocity, cfg.push_max_velocity, 2)

        if self.substep_callback is None:
            mujoco.mj_step(m, d, nstep=self.frame_skip)
        else:  # identical physics, one step at a time
            for _ in range(self.frame_skip):
                mujoco.mj_step(m, d)
                self.substep_callback(self)
        self._step_count += 1

        state = self._base_state()
        contacts, collisions, torso_contact = self._contact_state()
        terms = self._compute_reward_terms(state, action, contacts, collisions)

        reward = 0.0
        for name, scale in self._reward_terms.items():
            value = scale * terms[name]
            self._episode_sums[name] += value
            reward += value
        if cfg.only_positive_rewards:
            reward = max(reward, 0.0)

        proj_g, height = state[3], state[4]
        terminated = bool(
            torso_contact or height < cfg.min_base_height or -proj_g[2] < cfg.min_uprightness
        )

        self._last_action[:] = action
        if self._step_count % self._max_steps_per_command == 0:
            self._resample_command()

        obs = self._get_obs(state)
        self.foot_contacts = contacts
        info = {
            "x_velocity": float(state[1][0]),
            "y_velocity": float(state[1][1]),
            "yaw_rate": float(state[2][2]),
            "base_height": height,
            "command_vx": float(self.command[0]),
            "command_yaw": float(self.command[2]),
            "x_position": float(d.qpos[0]),
        }
        if terminated:
            info["episode_reward_terms"] = dict(self._episode_sums)
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, False, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("render() called without render_mode; pass render_mode='rgb_array'.")
            return None
        if self.render_mode == "human":
            return self._render_human()
        return self.render_frame()

    def render_frame(self, width: int | None = None, height: int | None = None) -> np.ndarray:
        """Render an RGB frame from a camera that follows the robot."""
        w = width or self.cfg.render_width
        h = height or self.cfg.render_height
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (w, h):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
            self._camera = mujoco.MjvCamera()
            self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._camera.distance = self.cfg.camera_distance
            self._camera.azimuth = self.cfg.camera_azimuth
            self._camera.elevation = self.cfg.camera_elevation
            self._camera.lookat[:] = self.data.xpos[self._torso_id]
        torso = self.data.xpos[self._torso_id]
        # Critically-damped follow keeps the video smooth while staying on the robot.
        self._camera.lookat[:2] += 0.25 * (torso[:2] - self._camera.lookat[:2])
        self._camera.lookat[2] = 0.22
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------ helpers
    def set_command(self, vx: float, vy: float = 0.0, yaw_rate: float = 0.0) -> None:
        """Pin the command (e.g. for evaluation or teleoperation)."""
        self._fixed_command = np.array([vx, vy, yaw_rate], dtype=float)
        self.command[:] = self._fixed_command

    def _resample_command(self) -> None:
        if self._fixed_command is not None:
            self.command[:] = self._fixed_command
            return
        cfg, rng = self.cfg, self.np_random
        self.command[:] = (
            rng.uniform(*cfg.command_vx_range),
            rng.uniform(*cfg.command_vy_range),
            rng.uniform(*cfg.command_yaw_range),
        )

    def _randomize_dynamics(self) -> None:
        m, rng, cfg = self.model, self.np_random, self.cfg
        m.geom_friction[self._friction_geoms, 0] = rng.uniform(*cfg.friction_range)
        m.body_mass[self._torso_id] = self._nominal_torso_mass + rng.uniform(*cfg.payload_range)
        strength = rng.uniform(*cfg.motor_strength_range, size=m.nu)
        m.actuator_gainprm[:, 0] = self._nominal_gain * strength
        m.actuator_biasprm[:, 1] = self._nominal_bias[:, 1] * strength
        mujoco.mj_setConst(m, self.data)

    def _base_state(self):
        d = self.data
        rot = d.xmat[self._torso_id].reshape(3, 3)  # world_R_body
        lin_vel_b = rot.T @ d.qvel[0:3]
        ang_vel_b = d.qvel[3:6].copy()  # free-joint angular velocity is already in body frame
        proj_g = -rot[2, :].copy()  # gravity direction expressed in the body frame
        return rot, lin_vel_b, ang_vel_b, proj_g, float(d.qpos[2])

    def _contact_state(self):
        d = self.data
        contacts = d.sensordata[self._touch_adr] > 1.0
        ncon = d.ncon
        if ncon == 0:
            return contacts, 0, False
        geoms = d.contact.geom[:ncon]
        # Robot geoms never collide with each other, so every contact involves the floor.
        robot_geoms = geoms[:, 0] + geoms[:, 1] - self._floor_geom
        collisions = int(np.count_nonzero(self._non_foot_lut[robot_geoms]))
        torso_contact = bool((robot_geoms == self._torso_geom).any())
        return contacts, collisions, torso_contact

    def _compute_reward_terms(self, state, action, contacts, collisions) -> dict[str, float]:
        cfg, d, dt = self.cfg, self.data, self.dt
        _, v, w, g, height = state
        q = d.qpos[self._qs]
        qd = d.qvel[self._vs]
        cmd = self.command

        lin_err = (cmd[0] - v[0]) ** 2 + (cmd[1] - v[1]) ** 2
        ang_err = (cmd[2] - w[2]) ** 2

        # Feet air time (legged_gym): reward steps that land after a long swing.
        contact_filt = contacts | self._last_contacts
        self._last_contacts[:] = contacts
        first_contact = (self._feet_air_time > 0.0) & contact_filt
        self._feet_air_time += dt
        if math.hypot(cmd[0], cmd[1]) > 0.1:
            air_time_reward = float((self._feet_air_time - cfg.feet_air_time_target) @ first_contact)
        else:
            air_time_reward = 0.0
        self._feet_air_time *= ~contact_filt

        foot_xy = d.site_xpos[self._foot_sites, :2]
        foot_vel = (foot_xy - self._last_foot_pos[:, :2]) / dt
        self._last_foot_pos[:, :2] = foot_xy
        slip = float(np.einsum("ij,ij->i", foot_vel, foot_vel) @ contacts)

        dof_acc = (qd - self._last_dof_vel) / dt
        self._last_dof_vel[:] = qd
        tau = d.actuator_force
        da = action - self._last_action
        limit_violation = np.maximum(self._soft_lo - q, 0.0) + np.maximum(q - self._soft_hi, 0.0)
        abad = q[self._abad_idx] - self.default_joint_pos[self._abad_idx]

        return {
            "tracking_lin_vel": math.exp(-lin_err / cfg.tracking_sigma),
            "tracking_ang_vel": math.exp(-ang_err / cfg.tracking_sigma),
            "lin_vel_z": v[2] * v[2],
            "ang_vel_xy": w[0] * w[0] + w[1] * w[1],
            "orientation": g[0] * g[0] + g[1] * g[1],
            "base_height": (height - cfg.base_height_target) ** 2,
            "torques": float(tau @ tau),
            "dof_acc": float(dof_acc @ dof_acc),
            "action_rate": float(da @ da),
            "feet_air_time": air_time_reward,
            "feet_slip": slip,
            "collision": float(collisions),
            "dof_pos_limits": float(limit_violation.sum()),
            "abad_deviation": float(abad @ abad),
        }

    def _get_obs(self, state) -> np.ndarray:
        cfg, d = self.cfg, self.data
        _, lin_vel_b, ang_vel_b, proj_g, _ = state
        parts = [proj_g]
        if cfg.include_base_lin_vel:
            parts.append(lin_vel_b * _LIN_VEL_SCALE)
        parts += [
            ang_vel_b * _ANG_VEL_SCALE,
            self.command * _COMMAND_SCALE,
            d.qpos[self._qs] - self.default_joint_pos,
            d.qvel[self._vs] * _DOF_VEL_SCALE,
            self._last_action,
        ]
        obs = np.concatenate(parts)
        if cfg.obs_noise_scale > 0.0:
            obs = obs + self.np_random.uniform(-1.0, 1.0, obs.shape) * cfg.obs_noise_scale * 0.05
        return obs.astype(np.float32)

    def _render_human(self):
        if self._viewer is None:
            import mujoco.viewer  # requires a display

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()
        return None
