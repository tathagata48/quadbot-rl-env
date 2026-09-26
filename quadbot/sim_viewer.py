"""Interactive 3D viewer for the QuadBot simulation (MuJoCo physics, drawn with three.js).

The physics and the policy run in Python on the real :class:`~quadbot.env.QuadBotEnv`; the
browser only draws every body at the pose MuJoCo computed, so this shows the simulation itself
rather than a pre-rendered video.

* ``mode="live"`` (Google Colab): the simulation advances in real time inside the notebook
  kernel while you steer the robot with the on-screen controls or the keyboard.
* ``mode="replay"`` (any notebook, or a standalone HTML file): a command schedule is simulated
  first and then played back with pause, scrubbing and slow motion.

Both modes have an orbit/zoom/pan camera, an optional follow camera, foot-contact highlighting,
the robot's path on the floor, and arrows that show external pushes.
"""
from __future__ import annotations

import base64
import dataclasses
import io
import json
import uuid
from typing import Callable, Sequence

import mujoco
import numpy as np

from . import LEG_NAMES
from .env import QuadBotEnv, QuadBotEnvConfig

THREE_VERSION = "0.160.0"
COLAB_CALLBACK = "quadbot.sim_step"
Command = tuple[float, float, float]
DEFAULT_TOUR: tuple[tuple[float, Command], ...] = (
    (4.0, (0.5, 0.0, 0.0)),
    (5.0, (0.8, 0.0, 0.5)),
    (4.0, (1.0, 0.0, 0.0)),
    (5.0, (0.7, 0.0, -0.5)),
    (4.0, (0.4, 0.0, 0.0)),
)
_FIELDS = ("t", "pose", "cmd", "vx", "wz", "h", "c", "cut", "push")
_LIVE_SESSIONS: dict[str, "SimulationRunner"] = {}


def _r(values, digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in np.ravel(values)]


# --------------------------------------------------------------------------- scene export
def _rgb_texture_id(model: mujoco.MjModel, mat_id: int) -> int:
    texid = np.atleast_1d(model.mat_texid[mat_id])
    role = int(getattr(getattr(mujoco, "mjtTextureRole", None), "mjTEXROLE_RGB", 1)) if texid.size > 1 else 0
    tex = int(texid[role])
    if tex < 0 or int(model.tex_type[tex]) != int(mujoco.mjtTexture.mjTEXTURE_2D):
        return -1
    return tex


def _texture_data_url(model: mujoco.MjModel, tex_id: int) -> str:
    from PIL import Image

    w, h = int(model.tex_width[tex_id]), int(model.tex_height[tex_id])
    channels = int(model.tex_nchannel[tex_id]) if hasattr(model, "tex_nchannel") else 3
    adr = int(model.tex_adr[tex_id])
    pixels = np.asarray(model.tex_data[adr : adr + w * h * channels]).reshape(h, w, channels)
    image = Image.fromarray(np.ascontiguousarray(pixels[..., :3]).astype(np.uint8), "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def scene_description(model: mujoco.MjModel) -> dict:
    """Visible geometry (MuJoCo's default geom groups 0-2) in a form the browser can draw."""
    feet = {}
    for k, leg in enumerate(LEG_NAMES):
        try:
            feet[model.geom(f"{leg}_foot").id] = k
        except KeyError:
            pass
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    geoms: list[dict] = []
    textures: list[dict] = []
    texture_index: dict[tuple[int, int], int] = {}
    for g in range(model.ngeom):
        if model.geom_group[g] > 2:  # collision-only geoms are hidden, as in MuJoCo's viewer
            continue
        mat = int(model.geom_matid[g])
        rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]
        if rgba[3] <= 0:
            continue
        item = dict(
            id=g,
            type=int(model.geom_type[g]),
            body=int(model.geom_bodyid[g]),
            size=_r(model.geom_size[g], 5),
            pos=_r(model.geom_pos[g], 5),
            quat=_r(model.geom_quat[g], 6),
            rgba=_r(rgba, 3),
            emission=round(float(model.mat_emission[mat]), 3) if mat >= 0 else 0.0,
            roughness=round(float(np.clip(1.0 - 0.7 * model.mat_shininess[mat], 0.3, 1.0)), 3) if mat >= 0 else 0.8,
            foot=feet.get(g, -1),
            texture=-1,
        )
        if item["type"] == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(model.geom_dataid[g])
            v0, nv = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            f0, nf = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            item["vert"] = _r(model.mesh_vert[v0 : v0 + nv], 5)
            item["face"] = [int(i) for i in np.ravel(model.mesh_face[f0 : f0 + nf])]
        if mat >= 0 and (tex := _rgb_texture_id(model, mat)) >= 0:
            if (tex, mat) not in texture_index:
                texture_index[(tex, mat)] = len(textures)
                textures.append(
                    dict(
                        png=_texture_data_url(model, tex),
                        repeat=_r(model.mat_texrepeat[mat], 4),
                        uniform=bool(model.mat_texuniform[mat]),
                    )
                )
            item["texture"] = texture_index[(tex, mat)]
        geoms.append(item)
    base = int(model.jnt_bodyid[free[0]]) if len(free) else 1
    return dict(nbody=int(model.nbody), base=base, geoms=geoms, textures=textures)


# ------------------------------------------------------------------------------ simulation
class SimulationRunner:
    """Steps the real QuadBot environment with a policy and packs body poses for the viewer.

    The environment is created without a time limit, so a live session can run indefinitely.
    Commands are clipped to the ranges the policy was trained on.
    """

    def __init__(
        self,
        policy: Callable[[np.ndarray], np.ndarray],
        env_cfg: QuadBotEnvConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.cfg = env_cfg or QuadBotEnvConfig()
        self.env = QuadBotEnv(config=self.cfg)
        self.policy = policy
        self.model, self.data, self.dt = self.env.model, self.env.data, self.env.dt
        self.rng = np.random.default_rng(seed)
        self.yaw_max = float(max(abs(v) for v in self.cfg.command_yaw_range))
        self.command: Command = (round(float(np.mean(self.cfg.command_vx_range)), 2), 0.0, 0.0)
        self.t, self.falls = 0.0, 0
        self._cut, self._push = True, 0
        self.obs, _ = self.env.reset(seed=seed, options={"command": self.command})

    def reset(self) -> None:
        self.obs, _ = self.env.reset(options={"command": self.command})
        self._cut = True

    def set_command(self, vx: float, yaw_rate: float = 0.0) -> None:
        vx = float(np.clip(vx, *self.cfg.command_vx_range))
        yaw_rate = float(np.clip(yaw_rate, -self.yaw_max, self.yaw_max))
        self.command = (vx, 0.0, yaw_rate)
        self.env.set_command(*self.command)

    def push(self, speed: float) -> None:
        """Sideways velocity kick on the torso, as if the robot were shoved."""
        w, x, y, z = self.data.qpos[3:7]
        heading = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        side = self.rng.choice((-1.0, 1.0))
        kick = speed * side * np.array([-np.sin(heading), np.cos(heading)])
        self.data.qvel[0:2] += kick
        self._push = _r(kick, 3)

    def run(self, n_steps: int, include_current: bool = False) -> dict:
        """Advance ``n_steps`` control steps; returns one viewer frame per step."""
        out: dict[str, list] = {key: [] for key in _FIELDS}
        if include_current:
            self._record(out, None)
        for _ in range(int(n_steps)):
            self.obs, _, terminated, _, info = self.env.step(self.policy(self.obs))
            self.t += self.dt
            self._record(out, info)
            if terminated:
                self.falls += 1
                self.reset()
        return out

    def _record(self, out: dict, info: dict | None) -> None:
        d = self.data
        mujoco.mj_kinematics(self.model, d)  # body poses for the current qpos
        out["t"].append(round(self.t, 4))
        out["pose"].append(_r(np.hstack([d.xpos[1:], d.xquat[1:]])))
        out["cmd"].append([round(self.command[0], 3), round(self.command[2], 3)])
        out["vx"].append(round(float(info["x_velocity"]), 3) if info else 0.0)
        out["wz"].append(round(float(info["yaw_rate"]), 3) if info else 0.0)
        out["h"].append(round(float(info["base_height"] if info else d.qpos[2]), 3))
        out["c"].append(int(np.dot(self.env.foot_contacts, (1, 2, 4, 8))))
        out["cut"].append(int(self._cut))
        out["push"].append(self._push)
        self._cut, self._push = False, 0

    def close(self) -> None:
        self.env.close()


def record_session(
    policy: Callable[[np.ndarray], np.ndarray],
    env_cfg: QuadBotEnvConfig | None = None,
    schedule: Sequence[tuple[float, Command]] = DEFAULT_TOUR,
    push_every: float = 0.0,
    push_speed: float = 0.5,
    seed: int = 0,
) -> dict:
    """Simulate ``schedule`` [(seconds, (vx, vy, yaw_rate)), ...] and return viewer frames.

    ``vy`` is ignored (the policy was trained to walk forward); ``push_every`` > 0 shoves the
    robot sideways every that many seconds.
    """
    runner = SimulationRunner(policy, env_cfg, seed)
    runner.set_command(schedule[0][1][0], schedule[0][1][2])
    runner.reset()
    frames = runner.run(0, include_current=True)
    since_push = 0.0
    for seconds, (vx, _vy, yaw_rate) in schedule:
        runner.set_command(vx, yaw_rate)
        for _ in range(int(round(seconds / runner.dt))):
            since_push += runner.dt
            if push_every > 0 and since_push >= push_every:
                runner.push(push_speed)
                since_push = 0.0
            for key, values in runner.run(1).items():
                frames[key].extend(values)
    frames["falls"] = runner.falls
    runner.close()
    return frames


# ----------------------------------------------------------------------------- live (Colab)
def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def _live_step(session: str, n_steps: int, vx: float, yaw_rate: float, push: float, reset: bool):
    """Colab callback: advance the live session and return the new frames as JSON."""
    from IPython.display import JSON

    runner = _LIVE_SESSIONS.get(session)
    if runner is None:
        return JSON({"error": "this live session has ended"})
    if reset:
        runner.reset()
    runner.set_command(float(vx), float(yaw_rate))
    if push:
        runner.push(float(push))
    frames = runner.run(int(np.clip(n_steps, 0, 100)), include_current=bool(reset))
    frames["falls"] = runner.falls
    return JSON(frames)


def _register_colab_callback() -> None:
    from google.colab import output

    output.register_callback(COLAB_CALLBACK, _live_step)


# ------------------------------------------------------------------------------ HTML output
def simulation_html(
    policy: Callable[[np.ndarray], np.ndarray],
    env_cfg: QuadBotEnvConfig | None = None,
    mode: str = "replay",
    schedule: Sequence[tuple[float, Command]] = DEFAULT_TOUR,
    push_every: float = 0.0,
    push_speed: float = 0.5,
    height: int = 560,
    seed: int = 0,
    domain_randomization: bool = False,
    standalone: bool = False,
) -> str:
    """HTML for the interactive viewer (see the module docstring for the two modes)."""
    if mode not in ("live", "replay"):
        raise ValueError("mode must be 'live' or 'replay'")
    cfg = dataclasses.replace(env_cfg or QuadBotEnvConfig(), domain_randomization=domain_randomization)
    uid = uuid.uuid4().hex[:12]
    runner = SimulationRunner(policy, cfg, seed)
    config = dict(
        mode=mode,
        session=uid,
        callback=COLAB_CALLBACK,
        vx_range=_r(cfg.command_vx_range, 3),
        yaw_max=round(runner.yaw_max, 3),
        vx0=runner.command[0],
        push_speed=push_speed,
        chunk=12,
        lead=0.35,
    )
    data = None
    scene = scene_description(runner.model)
    if mode == "replay":
        runner.close()
        data = record_session(policy, cfg, schedule, push_every, push_speed, seed)
    else:
        for old in _LIVE_SESSIONS.values():  # one live robot at a time
            old.close()
        _LIVE_SESSIONS.clear()
        _LIVE_SESSIONS[uid] = runner
        _register_colab_callback()
    page = (
        _TEMPLATE.replace("__UID__", uid)
        .replace("__HEIGHT__", str(int(height)))
        .replace("__THREE__", THREE_VERSION)
        .replace("__CONFIG__", json.dumps(config))
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__SCENE__", json.dumps(scene, separators=(",", ":")))
    )
    if standalone:
        page = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>QuadBot simulation</title></head>"
            "<body style='margin:0;padding:12px;background:#0a0e16'>" + page + "</body></html>"
        )
    return page


def show_simulation(
    policy: Callable[[np.ndarray], np.ndarray],
    env_cfg: QuadBotEnvConfig | None = None,
    mode: str = "auto",
    **kwargs,
) -> None:
    """Display the viewer in a notebook. ``mode="auto"``: live in Colab, replay elsewhere."""
    from IPython.display import HTML, display

    if mode == "auto":
        mode = "live" if in_colab() else "replay"
    if mode == "live" and not in_colab():
        raise RuntimeError("live mode needs Google Colab's kernel callbacks; use mode='replay' here")
    display(HTML(simulation_html(policy, env_cfg, mode=mode, **kwargs)))


def save_simulation_html(
    path: str,
    policy: Callable[[np.ndarray], np.ndarray],
    env_cfg: QuadBotEnvConfig | None = None,
    **kwargs,
) -> str:
    """Write a standalone replay page (it only needs internet access to load three.js)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(simulation_html(policy, env_cfg, mode="replay", standalone=True, **kwargs))
    return path


_TEMPLATE = r"""<div id="qb-__UID__" class="qb-root" tabindex="0">
  <div class="qb-msg">Loading the 3D viewer…</div>
  <div class="qb-panel qb-hud"></div>
  <div class="qb-panel qb-feet"><span>FL</span><span>FR</span><span>RL</span><span>RR</span></div>
  <div class="qb-panel qb-help"></div>
  <div class="qb-panel qb-bar"></div>
</div>
<style>
#qb-__UID__ { position: relative; width: 100%; height: __HEIGHT__px; min-height: 320px; background: #0d1424;
  border-radius: 10px; overflow: hidden; outline: none; color: #e8edf5;
  font: 13px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; user-select: none; -webkit-user-select: none; }
#qb-__UID__:focus-visible { box-shadow: inset 0 0 0 2px #f5851f; }
#qb-__UID__ canvas { display: block; cursor: grab; }
#qb-__UID__ .qb-panel { position: absolute; background: rgba(10, 14, 22, 0.74); border-radius: 8px; padding: 6px 10px; }
#qb-__UID__ .qb-hud { top: 10px; left: 10px; white-space: pre; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
#qb-__UID__ .qb-hud b { color: #f5851f; }
#qb-__UID__ .qb-feet { top: 10px; right: 10px; display: flex; gap: 5px; }
#qb-__UID__ .qb-feet span { padding: 2px 7px; border-radius: 5px; background: #2a3342; color: #8b96a8;
  font: 600 11px ui-monospace, Menlo, Consolas, monospace; }
#qb-__UID__ .qb-feet span.on { background: #16c8ff; color: #06202b; }
#qb-__UID__ .qb-help { top: 46px; right: 10px; font-size: 11px; color: #aab4c3; text-align: right; white-space: pre; }
#qb-__UID__ .qb-bar { left: 10px; right: 10px; bottom: 10px; display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; }
#qb-__UID__ .qb-bar label { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
#qb-__UID__ .qb-bar input[type=range] { accent-color: #f5851f; }
#qb-__UID__ .qb-grow { flex: 1 1 160px; }
#qb-__UID__ button, #qb-__UID__ select { background: #242d3b; color: #e8edf5; border: 1px solid #3a4557;
  border-radius: 6px; padding: 3px 10px; font: inherit; cursor: pointer; }
#qb-__UID__ button:hover { border-color: #f5851f; }
#qb-__UID__ .qb-num { display: inline-block; min-width: 5.5em; font-family: ui-monospace, Menlo, Consolas, monospace; }
#qb-__UID__ .qb-msg { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); max-width: 80%;
  padding: 10px 14px; border-radius: 8px; background: rgba(10, 14, 22, 0.85); text-align: center; pointer-events: none; z-index: 2; }
#qb-__UID__ .qb-msg:empty { display: none; }
</style>
<script>
setTimeout(function () {
  var root = document.getElementById("qb-__UID__");
  if (root && !root.dataset.ready) {
    root.querySelector(".qb-msg").textContent =
      "The 3D viewer did not start. It loads three.js from cdn.jsdelivr.net, so it needs internet access and WebGL.";
  }
}, 20000);
</script>
<script type="module">
import * as THREE from "https://cdn.jsdelivr.net/npm/three@__THREE__/+esm";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@__THREE__/examples/jsm/controls/OrbitControls.js/+esm";

const root = document.getElementById("qb-__UID__");
const SCENE = __SCENE__;
const CFG = __CONFIG__;
const REPLAY = __DATA__;
const LIVE = CFG.mode === "live";
const $ = (selector) => root.querySelector(selector);
const msg = (text) => { $(".qb-msg").textContent = text; };

// ---- core: geometry and poses, in MuJoCo's world frame (z up) ---------------------- @core-begin
const PLANE_HALF = 60;

function makeGeometry(g) {
  const s = g.size;
  switch (g.type) {
    case 0: return new THREE.PlaneGeometry(2 * (s[0] || PLANE_HALF), 2 * (s[1] || PLANE_HALF));
    case 2: return new THREE.SphereGeometry(s[0], 24, 16);
    case 3: return new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16).rotateX(Math.PI / 2);
    case 4: return new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]);
    case 5: return new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 32).rotateX(Math.PI / 2);
    case 6: return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    case 7: {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(g.vert, 3));
      geometry.setIndex(g.face);
      geometry.computeVertexNormals();
      return geometry;
    }
    default: return null; // height fields and SDF geoms are not drawn
  }
}

// One group per MuJoCo body (index = body id); geoms are children at their local pose.
function buildBodies(scene, makeMaterial) {
  const bodies = [];
  for (let b = 0; b < SCENE.nbody; b++) {
    const group = new THREE.Group();
    scene.add(group);
    bodies.push(group);
  }
  const feet = [], planes = [];
  SCENE.geoms.forEach((g, index) => {
    const geometry = makeGeometry(g);
    if (!geometry) return;
    const mesh = new THREE.Mesh(geometry, makeMaterial(g));
    mesh.position.fromArray(g.pos);
    mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
    mesh.castShadow = g.type !== 0;
    mesh.receiveShadow = true;
    mesh.userData.index = index;
    bodies[g.body].add(mesh);
    if (g.foot >= 0) feet[g.foot] = mesh;
    if (g.type === 0 && g.body === 0) planes.push({ mesh, g });
  });
  return { bodies, feet, planes };
}

const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion();
const _va = new THREE.Vector3(), _vb = new THREE.Vector3();

// Frames hold [x y z qw qx qy qz] for bodies 1..nbody-1; blend frame A towards B by alpha.
function setPose(bodies, A, B, alpha) {
  for (let i = 1; i < bodies.length; i++) {
    const o = 7 * (i - 1);
    _va.set(A[o], A[o + 1], A[o + 2]);
    _vb.set(B[o], B[o + 1], B[o + 2]);
    bodies[i].position.lerpVectors(_va, _vb, alpha);
    _qa.set(A[o + 4], A[o + 5], A[o + 6], A[o + 3]).normalize();
    _qb.set(B[o + 4], B[o + 5], B[o + 6], B[o + 3]).normalize();
    bodies[i].quaternion.slerpQuaternions(_qa, _qb, alpha);
  }
}
// ------------------------------------------------------------------------------------ @core-end

function start() {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true });
  } catch (err) {
    msg("WebGL is not available in this browser, so the 3D view cannot be shown.");
    return;
  }
  let simTime = 0, playing = true, speed = 1, seeking = false, follow = true, showTrail = true;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  root.prepend(renderer.domElement);

  // ---- world: sky, fog, camera, lights
  const scene = new THREE.Scene();
  const HORIZON = 0x3d5a78;
  scene.fog = new THREE.Fog(HORIZON, 6, 26);
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(100, 32, 16),
    new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      uniforms: { top: { value: new THREE.Color(0x0c1322) }, horizon: { value: new THREE.Color(HORIZON) } },
      vertexShader:
        "varying vec3 vDir;\nvoid main() { vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
      fragmentShader:
        "uniform vec3 top;\nuniform vec3 horizon;\nvarying vec3 vDir;\nvoid main() {\n  gl_FragColor = vec4(mix(horizon, top, smoothstep(-0.05, 0.5, vDir.z)), 1.0);\n#include <colorspace_fragment>\n}",
    })
  );
  sky.renderOrder = -1;
  scene.add(sky);

  const camera = new THREE.PerspectiveCamera(40, 1, 0.02, 300);
  camera.up.set(0, 0, 1);
  camera.position.set(1.3, -1.8, 0.75);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.22);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.3;
  controls.maxDistance = 20;
  controls.maxPolarAngle = Math.PI * 0.49;

  const hemi = new THREE.HemisphereLight(0xdde6f5, 0x2a2f38, 1.5);
  hemi.position.set(0, 0, 1);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff4e6, 2.6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2, near: 0.5, far: 20 });
  sun.shadow.camera.updateProjectionMatrix();
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.015;
  const SUN_OFFSET = new THREE.Vector3(-1.8, 2.4, 7.5);
  scene.add(sun, sun.target);
  const fill = new THREE.DirectionalLight(0xbfd0ff, 0.7);
  fill.position.set(2, -3, 2);
  scene.add(fill);

  // ---- robot and floor from the MuJoCo model
  const loader = new THREE.TextureLoader();
  const maxAnisotropy = renderer.capabilities.getMaxAnisotropy();
  function makeMaterial(g) {
    const color = new THREE.Color().setRGB(g.rgba[0], g.rgba[1], g.rgba[2], THREE.SRGBColorSpace);
    const params = { color, roughness: g.roughness, metalness: 0.05 };
    if (g.rgba[3] < 1) Object.assign(params, { transparent: true, opacity: g.rgba[3] });
    if (g.emission > 0) Object.assign(params, { emissive: color.clone(), emissiveIntensity: g.emission });
    if (g.foot >= 0) Object.assign(params, { emissive: new THREE.Color(0x000000), emissiveIntensity: 1.4 });
    if (g.texture >= 0) {
      const t = SCENE.textures[g.texture];
      const map = loader.load(t.png);
      map.colorSpace = THREE.SRGBColorSpace;
      map.wrapS = map.wrapT = THREE.RepeatWrapping;
      map.anisotropy = maxAnisotropy;
      const sx = 2 * (g.size[0] || PLANE_HALF), sy = 2 * (g.size[1] || PLANE_HALF);
      map.repeat.set(t.uniform ? t.repeat[0] * sx : t.repeat[0], t.uniform ? t.repeat[1] * sy : t.repeat[1]);
      params.map = map;
    }
    return new THREE.MeshStandardMaterial(params);
  }
  const { bodies, feet, planes } = buildBodies(scene, makeMaterial);
  const base = bodies[SCENE.base];

  const TRAIL_MAX = 2000;
  const trailPositions = new Float32Array(TRAIL_MAX * 3);
  const trailGeometry = new THREE.BufferGeometry();
  trailGeometry.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
  trailGeometry.setDrawRange(0, 0);
  const trail = new THREE.Line(trailGeometry, new THREE.LineBasicMaterial({ color: 0xf5851f, transparent: true, opacity: 0.85 }));
  trail.frustumCulled = false;
  scene.add(trail);
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.5, 0xff4d4d, 0.12, 0.07);
  arrow.visible = false;
  scene.add(arrow);

  // ---- frame buffer (replay: whole recording; live: rolling window)
  const F = { t: [], pose: [], cmd: [], vx: [], wz: [], h: [], c: [], cut: [], push: [], lastCut: [], lastPush: [], falls: [] };
  const KEYS = ["t", "pose", "cmd", "vx", "wz", "h", "c", "cut", "push"];
  const MAX_FRAMES = LIVE ? 4000 : Infinity;
  let cursor = 0, trailK = -1;
  function appendFrames(d) {
    for (let i = 0; i < d.t.length; i++) {
      const k = F.t.length;
      for (const key of KEYS) F[key].push(d[key][i]);
      F.lastCut.push(k === 0 || d.cut[i] ? k : F.lastCut[k - 1]);
      F.lastPush.push(d.push[i] ? k : k ? F.lastPush[k - 1] : -1);
      F.falls.push((k ? F.falls[k - 1] : 0) + (k > 0 && d.cut[i] ? 1 : 0));
    }
    const drop = F.t.length - MAX_FRAMES;
    if (drop > 0) {
      for (const key of Object.keys(F)) F[key].splice(0, drop);
      F.lastCut = F.lastCut.map((v) => Math.max(0, v - drop));
      F.lastPush = F.lastPush.map((v) => (v >= drop ? v - drop : -1));
      cursor = Math.max(0, cursor - drop);
      trailK = -1;
    }
  }
  function clearFrames() {
    for (const key of Object.keys(F)) F[key].length = 0;
    cursor = 0;
    trailK = -1;
  }
  function frameAt(time) {
    const n = F.t.length;
    if (!n) return -1;
    cursor = Math.min(cursor, n - 1);
    while (cursor + 1 < n && F.t[cursor + 1] <= time) cursor++;
    while (cursor > 0 && F.t[cursor] > time) cursor--;
    return cursor;
  }
  function applyTime(time) {
    const k = frameAt(time);
    if (k < 0) return -1;
    let next = F.pose[k], alpha = 0;
    if (k + 1 < F.t.length && !F.cut[k + 1]) {
      next = F.pose[k + 1];
      alpha = THREE.MathUtils.clamp((time - F.t[k]) / (F.t[k + 1] - F.t[k]), 0, 1);
    }
    setPose(bodies, F.pose[k], next, alpha);
    return k;
  }

  // ---- per-frame updates
  const _desired = new THREE.Vector3(), _delta = new THREE.Vector3(), _dir = new THREE.Vector3();
  function updateCamera(dt) {
    const p = base.position;
    if (follow) {
      _desired.set(p.x, p.y, controls.target.z);
      _delta.subVectors(_desired, controls.target);
      if (_delta.length() < 3) _delta.multiplyScalar(1 - Math.exp(-6 * dt));
      controls.target.add(_delta);
      camera.position.add(_delta);
    }
    sun.target.position.copy(p);
    sun.position.copy(p).add(SUN_OFFSET);
    for (const { mesh, g } of planes) {
      if (g.size[0] || g.size[1]) continue; // only infinite planes follow the robot
      const map = mesh.material.map;
      const px = map ? (2 * PLANE_HALF) / map.repeat.x : 1;
      const py = map ? (2 * PLANE_HALF) / map.repeat.y : 1;
      mesh.position.x = g.pos[0] + Math.round((p.x - g.pos[0]) / px) * px;
      mesh.position.y = g.pos[1] + Math.round((p.y - g.pos[1]) / py) * py;
    }
  }
  function updateTrail(k) {
    trail.visible = showTrail;
    if (k === trailK) return;
    trailK = k;
    const start = Math.max(F.lastCut[k], k - TRAIL_MAX + 1);
    const o = 7 * (SCENE.base - 1);
    let n = 0;
    for (let j = start; j <= k; j++, n++) {
      trailPositions[3 * n] = F.pose[j][o];
      trailPositions[3 * n + 1] = F.pose[j][o + 1];
      trailPositions[3 * n + 2] = 0.004;
    }
    trailGeometry.setDrawRange(0, n);
    trailGeometry.attributes.position.needsUpdate = true;
  }
  function updatePush(k, time) {
    const j = F.lastPush[k];
    if (j < 0 || time - F.t[j] > 0.9) {
      arrow.visible = false;
      return;
    }
    const v = F.push[j];
    _dir.set(v[0], v[1], 0);
    const length = 0.25 + 0.5 * _dir.length();
    _dir.normalize();
    arrow.position.set(base.position.x - _dir.x * (length + 0.12), base.position.y - _dir.y * (length + 0.12), base.position.z + 0.05);
    arrow.setDirection(_dir);
    arrow.setLength(length, 0.12, 0.07);
    arrow.visible = true;
  }
  const hudEl = $(".qb-hud");
  const footEls = [...root.querySelectorAll(".qb-feet span")];
  const fmt = (v) => (v >= 0 ? "+" : "") + v.toFixed(2);
  let liveFalls = 0, latency = 0;
  function updateFeet(k) {
    for (let i = 0; i < 4; i++) {
      const on = ((F.c[k] >> i) & 1) === 1;
      footEls[i].classList.toggle("on", on);
      if (feet[i]) feet[i].material.emissive.setHex(on ? 0x16c8ff : 0x000000);
    }
  }
  function updateHud(k) {
    const [cvx, cwz] = F.cmd[k];
    const head = LIVE
      ? `<b>LIVE SIMULATION</b>   t ${F.t[k].toFixed(2)} s   kernel ${latency.toFixed(0)} ms`
      : `<b>REPLAY</b>   t ${(F.t[k] - F.t[0]).toFixed(2)} / ${(F.t[F.t.length - 1] - F.t[0]).toFixed(1)} s`;
    hudEl.innerHTML =
      `${head}\ncommand   vx ${fmt(cvx)} m/s   yaw ${fmt(cwz)} rad/s\n` +
      `measured  vx ${fmt(F.vx[k])} m/s   yaw ${fmt(F.wz[k])} rad/s\n` +
      `height ${F.h[k].toFixed(3)} m   falls ${LIVE ? liveFalls : F.falls[k]}`;
  }

  // ---- controls
  const bar = $(".qb-bar");
  const cmd = { vx: CFG.vx0, yaw: 0 };
  let pushReq = 0, resetReq = LIVE;
  if (LIVE) {
    bar.innerHTML = `
      <button data-act="play">pause</button>
      <label>speed <input data-act="vx" type="range" min="${CFG.vx_range[0]}" max="${CFG.vx_range[1]}" step="0.05" value="${CFG.vx0}"><span class="qb-num" data-out="vx"></span></label>
      <label>turn <input data-act="yaw" type="range" min="${-CFG.yaw_max}" max="${CFG.yaw_max}" step="0.05" value="0" ${CFG.yaw_max ? "" : "disabled"}><span class="qb-num" data-out="yaw"></span></label>
      <button data-act="push">push</button>
      <button data-act="reset">reset</button>
      <label><input data-act="follow" type="checkbox" checked> follow</label>
      <label><input data-act="trail" type="checkbox" checked> path</label>`;
    $(".qb-help").textContent = "drag: orbit · right-drag: pan · wheel: zoom\nkeys (click the view first): arrows = speed / turn · space = push · R = reset";
  } else {
    const speeds = [0.1, 0.25, 0.5, 1, 2].map((v) => `<option value="${v}"${v === 1 ? " selected" : ""}>${v}×</option>`);
    bar.innerHTML = `
      <button data-act="play">pause</button>
      <select data-act="speed">${speeds.join("")}</select>
      <input class="qb-grow" data-act="seek" type="range" min="0" max="1000" value="0">
      <label><input data-act="follow" type="checkbox" checked> follow</label>
      <label><input data-act="trail" type="checkbox" checked> path</label>`;
    $(".qb-help").textContent = "drag: orbit · right-drag: pan · wheel: zoom\nkeys (click the view first): space = play / pause · left / right = skip 1 s";
  }
  const el = (act) => bar.querySelector(`[data-act="${act}"]`);
  const out = (name) => bar.querySelector(`[data-out="${name}"]`);
  function syncCommand() {
    el("vx").value = cmd.vx;
    el("yaw").value = cmd.yaw;
    out("vx").textContent = `${cmd.vx.toFixed(2)} m/s`;
    out("yaw").textContent = `${fmt(cmd.yaw)} rad/s`;
  }
  function setPlaying(value) {
    playing = value;
    el("play").textContent = playing ? "pause" : "play";
  }
  el("play").addEventListener("click", () => setPlaying(!playing));
  el("follow").addEventListener("change", (e) => { follow = e.target.checked; });
  el("trail").addEventListener("change", (e) => { showTrail = e.target.checked; });
  const span = () => F.t[F.t.length - 1] - F.t[0];
  if (LIVE) {
    el("vx").addEventListener("input", (e) => { cmd.vx = +e.target.value; syncCommand(); });
    el("yaw").addEventListener("input", (e) => { cmd.yaw = +e.target.value; syncCommand(); });
    el("push").addEventListener("click", () => { pushReq = CFG.push_speed; });
    el("reset").addEventListener("click", () => { resetReq = true; });
    syncCommand();
  } else {
    const seek = el("seek");
    el("speed").addEventListener("change", (e) => { speed = +e.target.value; });
    seek.addEventListener("pointerdown", () => { seeking = true; });
    window.addEventListener("pointerup", () => { seeking = false; });
    seek.addEventListener("input", () => { simTime = F.t[0] + (seek.value / 1000) * span(); });
  }
  root.addEventListener("pointerdown", () => root.focus({ preventScroll: true }));
  root.addEventListener("keydown", (e) => {
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    const step = (value, delta, lo, hi) => Math.min(hi, Math.max(lo, Math.round((value + delta) * 100) / 100));
    let handled = true;
    if (LIVE) {
      if (key === "ArrowUp") cmd.vx = step(cmd.vx, 0.05, CFG.vx_range[0], CFG.vx_range[1]);
      else if (key === "ArrowDown") cmd.vx = step(cmd.vx, -0.05, CFG.vx_range[0], CFG.vx_range[1]);
      else if (key === "ArrowLeft") cmd.yaw = step(cmd.yaw, 0.1, -CFG.yaw_max, CFG.yaw_max);
      else if (key === "ArrowRight") cmd.yaw = step(cmd.yaw, -0.1, -CFG.yaw_max, CFG.yaw_max);
      else if (key === " ") pushReq = CFG.push_speed;
      else if (key === "r") resetReq = true;
      else if (key === "p") setPlaying(!playing);
      else handled = false;
      syncCommand();
    } else if (key === " ") setPlaying(!playing);
    else if (key === "ArrowLeft") simTime = Math.max(F.t[0], simTime - 1);
    else if (key === "ArrowRight") simTime = Math.min(F.t[0] + span(), simTime + 1);
    else handled = false;
    if (handled) e.preventDefault();
  });

  // ---- live mode: the Colab kernel advances the physics in small chunks
  let pending = false, pendingSince = 0;
  async function pump() {
    if (pending) return;
    const kernel = window.google && window.google.colab && window.google.colab.kernel;
    if (!kernel) {
      msg("Live mode runs inside Google Colab. Use mode='replay' in other notebooks.");
      return;
    }
    const ahead = F.t.length ? F.t[F.t.length - 1] - simTime : 0;
    const reset = resetReq, push = pushReq;
    if (!reset && !push && !(playing && ahead < CFG.lead)) return;
    resetReq = false;
    pushReq = 0;
    pending = true;
    pendingSince = performance.now();
    try {
      const n = reset ? 0 : CFG.chunk;
      const result = await kernel.invokeFunction(CFG.callback, [CFG.session, n, cmd.vx, cmd.yaw, push, reset], {});
      const d = result.data["application/json"];
      if (d.error) throw new Error(d.error);
      if (reset) {
        clearFrames();
        simTime = d.t[0];
      }
      appendFrames(d);
      liveFalls = d.falls;
      latency = performance.now() - pendingSince;
      msg("");
    } catch (err) {
      setPlaying(false);
      msg(`Live simulation stopped (${err && err.message ? err.message : err}). Re-run the cell to start a new session.`);
    } finally {
      pending = false;
    }
  }

  // ---- main loop
  if (!LIVE) {
    if (!REPLAY || !REPLAY.t.length) {
      msg("No simulation data to show.");
      return;
    }
    appendFrames(REPLAY);
    simTime = F.t[0];
  } else {
    setInterval(pump, 30);
  }
  const clock = new THREE.Clock();
  let hudClock = 1;
  function tick() {
    if (!root.isConnected) {
      renderer.setAnimationLoop(null);
      renderer.dispose();
      return;
    }
    const dt = Math.min(clock.getDelta(), 0.1);
    if (F.t.length) {
      const t0 = F.t[0], t1 = F.t[F.t.length - 1];
      if (playing && !seeking) simTime += dt * speed;
      if (LIVE) simTime = THREE.MathUtils.clamp(simTime, t0, t1);
      else if (simTime > t1) simTime = t0; // loop the replay
      const k = applyTime(simTime);
      updateCamera(dt);
      updateTrail(k);
      updatePush(k, simTime);
      updateFeet(k);
      hudClock += dt;
      if (hudClock > 0.08) {
        hudClock = 0;
        updateHud(k);
        if (!LIVE && !seeking) el("seek").value = Math.round((1000 * (simTime - t0)) / Math.max(t1 - t0, 1e-6));
      }
    }
    if (LIVE && pending && performance.now() - pendingSince > 2000) {
      msg("Waiting for the Colab kernel… (it cannot run the simulation while another cell is running)");
    }
    controls.update();
    sky.position.copy(camera.position);
    renderer.render(scene, camera);
  }
  function resize() {
    const w = Math.max(root.clientWidth, 1), h = Math.max(root.clientHeight, 1);
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(root);
  resize();
  root.dataset.ready = "1";
  msg(LIVE ? "Starting the live simulation…" : "");
  renderer.setAnimationLoop(tick);
}

try {
  start();
} catch (err) {
  console.error(err);
  msg(`3D viewer error: ${err.message}`);
}
</script>
"""
