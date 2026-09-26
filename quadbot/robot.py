"""QuadBot: a custom 12-DoF quadruped described in MJCF.

The model is generated from a :class:`RobotSpec` so that leg geometry, mass
distribution and motor parameters can be changed from Python (useful for
design studies and domain randomisation) without hand-editing four copies of
the same leg in XML.

Kinematic layout (per leg, all legs identical, mirrored left/right):

    torso --[abad: x-axis]--> hip link --[hip: y-axis]--> thigh
          --[knee: y-axis]--> calf --> foot (sphere, the only intended contact)

Leg order everywhere in this project: FL, FR, RL, RR.
Joint order per leg: abad, hip, knee.
"""
from __future__ import annotations

from dataclasses import dataclass

LEG_NAMES: tuple[str, ...] = ("FL", "FR", "RL", "RR")
JOINT_SUFFIXES: tuple[str, ...] = ("abad", "hip", "knee")
JOINT_NAMES: tuple[str, ...] = tuple(f"{leg}_{j}" for leg in LEG_NAMES for j in JOINT_SUFFIXES)
FOOT_GEOM_NAMES: tuple[str, ...] = tuple(f"{leg}_foot" for leg in LEG_NAMES)


@dataclass(frozen=True)
class RobotSpec:
    """Physical parameters of QuadBot (SI units, radians)."""

    # Torso (full box dimensions) and mass.
    torso_size: tuple[float, float, float] = (0.38, 0.18, 0.09)
    torso_mass: float = 5.5
    # Abduction-axis position relative to the torso centre (x forward, y left).
    hip_offset_x: float = 0.18
    hip_offset_y: float = 0.055
    # Lateral offset from the abduction axis to the thigh (hip pitch) joint.
    hip_link_length: float = 0.07
    thigh_length: float = 0.20
    calf_length: float = 0.20
    foot_radius: float = 0.022
    hip_mass: float = 0.55
    thigh_mass: float = 0.70
    calf_mass: float = 0.16
    foot_mass: float = 0.04
    # Joint limits.
    abad_range: tuple[float, float] = (-0.80, 0.80)
    hip_range: tuple[float, float] = (-1.00, 2.60)
    knee_range: tuple[float, float] = (-2.70, -0.70)
    # Motors: joint-level PD servo (MuJoCo <position> actuator).
    kp: float = 50.0
    kd: float = 1.5
    abad_torque_limit: float = 25.0
    hip_torque_limit: float = 25.0
    knee_torque_limit: float = 35.0
    joint_armature: float = 0.012  # reflected rotor inertia
    joint_damping: float = 0.05
    joint_frictionloss: float = 0.05
    foot_friction: float = 1.0
    # Nominal standing posture per leg: (abad, hip, knee).
    default_joint_angles: tuple[float, float, float] = (0.0, 0.8, -1.5)

    @property
    def total_mass(self) -> float:
        per_leg = self.hip_mass + self.thigh_mass + self.calf_mass + self.foot_mass
        return self.torso_mass + 4.0 * per_leg

    @property
    def nominal_height(self) -> float:
        """Torso-centre height when standing in the default posture."""
        import math

        _, q_hip, q_knee = self.default_joint_angles
        return (
            self.thigh_length * math.cos(q_hip)
            + self.calf_length * math.cos(q_hip + q_knee)
            + self.foot_radius
        )


_LEG_TEMPLATE = """
      <body name="{leg}_hip" pos="{hx:.4f} {hy:.4f} 0">
        <joint name="{leg}_abad" class="abad"/>
        <geom class="visual" type="cylinder" size="0.042 0.030" pos="0 0 0" euler="0 1.5708 0" material="motor"/>
        <geom class="collision" type="cylinder" size="0.040 0.028" pos="0 {ly_half:.4f} 0" euler="1.5708 0 0" mass="{hip_mass:.4f}"/>
        <geom class="visual" type="cylinder" size="0.040 0.028" pos="0 {ly_half:.4f} 0" euler="1.5708 0 0" material="accent"/>
        <body name="{leg}_thigh" pos="0 {ly:.4f} 0">
          <joint name="{leg}_hip" class="hip"/>
          <geom class="collision" type="capsule" fromto="0 0 0 0 0 {thigh_end:.4f}" size="0.022" mass="{thigh_mass:.4f}"/>
          <geom class="visual" type="capsule" fromto="0 0 0 0 0 {thigh_end:.4f}" size="0.024" material="limb"/>
          <geom class="visual" type="cylinder" size="0.036 0.022" euler="1.5708 0 0" material="motor"/>
          <body name="{leg}_calf" pos="0 0 {thigh_end:.4f}">
            <joint name="{leg}_knee" class="knee"/>
            <geom class="visual" type="cylinder" size="0.026 0.020" euler="1.5708 0 0" material="accent"/>
            <geom class="collision" type="capsule" fromto="0 0 0 0 0 {calf_geom_end:.4f}" size="0.014" mass="{calf_mass:.4f}"/>
            <geom class="visual" type="capsule" fromto="0 0 0 0 0 {calf_geom_end:.4f}" size="0.015" material="limb_dark"/>
            <geom name="{leg}_foot" class="foot" pos="0 0 {calf_end:.4f}" size="{foot_r:.4f}" mass="{foot_mass:.4f}"/>
            <site name="{leg}_foot_site" pos="0 0 {calf_end:.4f}" size="{site_r:.4f}" rgba="0 0 0 0"/>
          </body>
        </body>
      </body>"""


def _leg_xml(spec: RobotSpec, leg: str) -> str:
    sx = 1.0 if leg[0] == "F" else -1.0
    sy = 1.0 if leg[1] == "L" else -1.0
    return _LEG_TEMPLATE.format(
        leg=leg,
        hx=sx * spec.hip_offset_x,
        hy=sy * spec.hip_offset_y,
        ly=sy * spec.hip_link_length,
        ly_half=sy * 0.5 * spec.hip_link_length,
        hip_mass=spec.hip_mass,
        thigh_mass=spec.thigh_mass,
        calf_mass=spec.calf_mass,
        foot_mass=spec.foot_mass,
        thigh_end=-spec.thigh_length,
        calf_end=-spec.calf_length,
        calf_geom_end=-(spec.calf_length - spec.foot_radius),
        foot_r=spec.foot_radius,
        site_r=spec.foot_radius * 1.25,
    )


def build_quadbot_xml(spec: RobotSpec | None = None, sim_timestep: float = 0.004) -> str:
    """Return a complete MJCF document (robot + flat ground + lights)."""
    s = spec or RobotSpec()
    tx, ty, tz = (0.5 * v for v in s.torso_size)
    legs = "".join(_leg_xml(s, leg) for leg in LEG_NAMES)
    actuators = []
    for leg in LEG_NAMES:
        for joint, limit in zip(
            JOINT_SUFFIXES, (s.abad_torque_limit, s.hip_torque_limit, s.knee_torque_limit)
        ):
            lo, hi = getattr(s, f"{joint}_range")
            actuators.append(
                f'    <position name="{leg}_{joint}_servo" joint="{leg}_{joint}" kp="{s.kp}" '
                f'kv="{s.kd}" ctrlrange="{lo} {hi}" forcerange="{-limit} {limit}"/>'
            )
    touch = "\n".join(
        f'    <touch name="{leg}_touch" site="{leg}_foot_site"/>' for leg in LEG_NAMES
    )
    spawn_z = s.nominal_height + 0.02

    return f"""<mujoco model="quadbot">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{sim_timestep}" integrator="implicitfast" iterations="10" ls_iterations="10"/>

  <visual>
    <global offwidth="1920" offheight="1080" azimuth="135" elevation="-20"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <map znear="0.01" zfar="60"/>
  </visual>

  <default>
    <geom density="0"/>
    <default class="quadbot">
      <joint damping="{s.joint_damping}" armature="{s.joint_armature}" frictionloss="{s.joint_frictionloss}"/>
      <default class="abad"><joint axis="1 0 0" range="{s.abad_range[0]} {s.abad_range[1]}"/></default>
      <default class="hip"><joint axis="0 1 0" range="{s.hip_range[0]} {s.hip_range[1]}"/></default>
      <default class="knee"><joint axis="0 1 0" range="{s.knee_range[0]} {s.knee_range[1]}"/></default>
      <default class="visual"><geom contype="0" conaffinity="0" group="2"/></default>
      <!-- Robot collision geoms only collide with the environment, never with each other. -->
      <default class="collision"><geom contype="1" conaffinity="0" group="3" rgba="0.8 0.2 0.2 0.3"/></default>
      <default class="foot">
        <geom type="sphere" contype="1" conaffinity="0" group="2" condim="3"
              friction="{s.foot_friction} 0.02 0.01" material="rubber"/>
      </default>
    </default>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.42 0.60 0.80" rgb2="0.05 0.08 0.14" width="512" height="3072"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.20 0.24 0.29" rgb2="0.13 0.16 0.20"
             mark="edge" markrgb="0.45 0.55 0.65" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" reflectance="0.08"/>
    <material name="body" rgba="0.16 0.17 0.20 1" specular="0.4" shininess="0.6"/>
    <material name="shell" rgba="0.96 0.52 0.12 1" specular="0.5" shininess="0.7"/>
    <material name="accent" rgba="0.96 0.52 0.12 1" specular="0.5" shininess="0.6"/>
    <material name="motor" rgba="0.30 0.32 0.36 1" specular="0.6" shininess="0.8"/>
    <material name="limb" rgba="0.82 0.84 0.87 1" specular="0.4" shininess="0.5"/>
    <material name="limb_dark" rgba="0.22 0.23 0.26 1" specular="0.3" shininess="0.4"/>
    <material name="rubber" rgba="0.06 0.06 0.07 1" specular="0.1"/>
    <material name="lens" rgba="0.10 0.85 1.00 1" emission="0.9"/>
  </asset>

  <worldbody>
    <light name="fill" directional="true" pos="0 0 6" dir="0.4 -0.3 -1" diffuse="0.22 0.22 0.25" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="1" condim="3"/>

    <body name="torso" pos="0 0 {spawn_z:.4f}" childclass="quadbot">
      <freejoint name="root"/>
      <!-- Key light follows the robot so its shadow stays sharp wherever it walks. -->
      <light name="sun" mode="trackcom" directional="true" pos="-0.6 0.8 2.5" dir="0.24 -0.32 -1"
             diffuse="0.72 0.72 0.68" specular="0.2 0.2 0.2" castshadow="true"/>
      <camera name="side" mode="trackcom" pos="0 -1.4 0.35" xyaxes="1 0 0 0 0.25 1"/>
      <site name="imu" pos="0 0 0" size="0.01" rgba="0 0 0 0"/>
      <geom name="torso" class="collision" type="box" size="{tx:.4f} {ty:.4f} {tz:.4f}" mass="{s.torso_mass}"/>
      <geom class="visual" type="box" size="{tx:.4f} {ty:.4f} {tz:.4f}" material="body"/>
      <geom class="visual" type="box" size="{tx * 0.82:.4f} {ty * 0.86:.4f} 0.008" pos="0 0 {tz + 0.006:.4f}" material="shell"/>
      <geom class="visual" type="box" size="{tx * 0.40:.4f} {ty * 0.50:.4f} 0.014" pos="{-tx * 0.25:.4f} 0 {tz + 0.024:.4f}" material="motor"/>
      <!-- Sensor head -->
      <geom class="visual" type="box" size="0.035 {ty * 0.72:.4f} {tz * 0.62:.4f}" pos="{tx + 0.025:.4f} 0 0.008" material="limb_dark"/>
      <geom class="visual" type="cylinder" size="0.013 0.006" pos="{tx + 0.061:.4f} 0.032 0.012" euler="0 1.5708 0" material="lens"/>
      <geom class="visual" type="cylinder" size="0.013 0.006" pos="{tx + 0.061:.4f} -0.032 0.012" euler="0 1.5708 0" material="lens"/>
{legs}
    </body>
  </worldbody>

  <actuator>
{chr(10).join(actuators)}
  </actuator>

  <sensor>
{touch}
  </sensor>
</mujoco>
"""


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import mujoco

    spec = RobotSpec()
    model = mujoco.MjModel.from_xml_string(build_quadbot_xml(spec))
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
    print(f"total mass (spec) = {spec.total_mass:.2f} kg, model = {sum(model.body_mass):.2f} kg")
    print(f"nominal height = {spec.nominal_height:.3f} m")
