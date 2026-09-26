# -*- coding: utf-8 -*-
"""Regenerate every diagram in the README.

    python tools/render_diagrams.py

Writes a light and a dark SVG per diagram into assets/diagrams/. Pure stdlib --
the drawings are laid out from the same numbers the code uses (RobotSpec,
DEFAULT_REWARD_SCALES, TrainConfig, runs/main/results.json), so a change to the
robot or the reward is one command away from being a change to the pictures.
"""
import math
import os


FONT = "system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"

DARK = dict(name='dark', bg='#0d1117', card='#161b22', card2='#1b2230', border='#30363d',
            text='#e6edf3', muted='#9aa4b0', dim='#6e7681', wire='#4d5661', grid='#21262d',
            accent='#ff7a59', cyan='#58d4ff', green='#4ac26b', purple='#bc8cff',
            yellow='#e3b341', red='#f85149', metal='#c9d1d9')
LIGHT = dict(name='light', bg='#ffffff', card='#f6f8fa', card2='#eaeef2', border='#d0d7de',
             text='#1f2328', muted='#59636e', dim='#818b98', wire='#9aa4ae', grid='#eaeef2',
             accent='#c4441c', cyan='#0969da', green='#1a7f37', purple='#8250df',
             yellow='#9a6700', red='#cf222e', metal='#57606a')
THEMES = (DARK, LIGHT)


def esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def tx(x, y, s, size=12, fill='#000', anchor='start', weight=400, mono=False, op=None, ls=None):
    f = MONO if mono else FONT
    a = (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{f}" font-size="{size}" fill="{fill}"'
         f' text-anchor="{anchor}" font-weight="{weight}"')
    if op is not None:
        a += f' opacity="{op}"'
    if ls:
        a += f' letter-spacing="{ls}"'
    return a + f'>{esc(s)}</text>'


def rr(x, y, w, h, r=10, fill='none', stroke='none', sw=1, dash=None, op=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    o = f' opacity="{op}"' if op is not None else ''
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}"'
            f' fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')


def circ(x, y, r, fill='none', stroke='none', sw=1, op=None):
    o = f' opacity="{op}"' if op is not None else ''
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}"'
            f' stroke="{stroke}" stroke-width="{sw}"{o}/>')


def path(d, stroke='none', sw=1.5, fill='none', dash=None, cap='round', op=None, join='round'):
    s = f' stroke-dasharray="{dash}"' if dash else ''
    o = f' opacity="{op}"' if op is not None else ''
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
            f' stroke-linecap="{cap}" stroke-linejoin="{join}"{s}{o}/>')


def line(x1, y1, x2, y2, c, sw=1.5, dash=None, cap='round', op=None):
    return path(f'M{x1:.1f} {y1:.1f}L{x2:.1f} {y2:.1f}', c, sw, dash=dash, cap=cap, op=op)


def head(x, y, ang, c, s=7):
    pts = [(x, y)] + [(x + s * math.cos(ang + da), y + s * math.sin(ang + da)) for da in (2.55, -2.55)]
    return '<polygon points="%s" fill="%s"/>' % (' '.join(f'{a:.1f},{b:.1f}' for a, b in pts), c)


def arrow(x1, y1, x2, y2, c, sw=1.5, dash=None, s=7, op=None):
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - s * 0.8 * math.cos(ang), y2 - s * 0.8 * math.sin(ang)
    return line(x1, y1, bx, by, c, sw, dash, op=op) + head(x2, y2, ang, c, s)


def elbow(pts, c, sw=1.5, dash=None, arrow_end=True, s=7):
    """Orthogonal polyline through pts, optionally arrow-headed at the last point."""
    d = 'M' + 'L'.join(f'{x:.1f} {y:.1f}' for x, y in pts)
    out = path(d, c, sw, dash=dash)
    if arrow_end:
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        ang = math.atan2(y2 - y1, x2 - x1)
        out += head(x2, y2, ang, c, s)
    return out


def chip(x, y, label, value, T, color=None, h=24, pad=9, mono_val=True):
    """Small pill: grey label + coloured value. Returns (svg, width)."""
    w = 5.5 * len(label) + 6.4 * len(str(value)) + 32
    s = rr(x, y, w, h, h / 2, T['card2'], T['border'], 1)
    s += tx(x + pad, y + h / 2 + 3.6, label, 10, T['muted'])
    s += tx(x + w - pad, y + h / 2 + 3.6, value, 10.5, color or T['text'], 'end', 600, mono=mono_val)
    return s, w


def svg(w, h, T, body, title):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d"'
            ' role="img" aria-label="%s"><title>%s</title>'
            '<rect x="0.5" y="0.5" width="%.1f" height="%.1f" rx="14" fill="%s" stroke="%s"/>%s</svg>'
            % (w, h, w, h, esc(title), esc(title), w - 1, h - 1, T['bg'], T['border'], body))

def d_pipeline(T):
    W, H = 960, 296
    b = [tx(W / 2, 32, 'FROM A DATACLASS TO A WALKING POLICY', 11.5, T['muted'], 'middle', 700, ls=1.8)]
    stages = [('RobotSpec', 'frozen dataclass', '0.20 m links · 11.3 kg', T['purple']),
              ('MJCF model', 'build_quadbot_xml()', '4 legs × 3 joints', T['purple']),
              ('QuadBot-v0', 'Gymnasium env', '48-D obs → 12-D act', T['cyan']),
              ('PPO', 'Stable-Baselines3', '16 envs × 8M steps', T['accent']),
              ('policy.npz', 'exported actor', '4 matmuls, no torch', T['green'])]
    bw, gap, ys, bh = 158, 34, 52, 92
    x0 = (W - (5 * bw + 4 * gap)) / 2
    for i, (t, s1, s2, c) in enumerate(stages):
        x = x0 + i * (bw + gap)
        b.append(rr(x, ys, bw, bh, 12, T['card'], T['border'], 1))
        b.append(rr(x + 16, ys + 15, 24, 4, 2, c))
        b.append(tx(x + 16, ys + 42, t, 14, T['text'], weight=700))
        b.append(tx(x + 16, ys + 61, s1, 10.5, c, mono=True))
        b.append(tx(x + 16, ys + 77, s2, 10.5, T['muted']))
        if i < 4:
            ax = x + bw
            b.append(arrow(ax + 6, ys + bh / 2, ax + gap - 6, ys + bh / 2, T['wire'], 1.6))
            b.append(tx(ax + gap / 2, ys + bh / 2 - 11, ['emit', 'wrap', 'train', 'export'][i],
                        9, T['dim'], 'middle', 600))
    # deployment bus
    npz_cx = x0 + 4 * (bw + gap) + bw / 2
    bus_y = 178
    cw, cgap = 290, 25
    cx0 = (W - (3 * cw + 2 * cgap)) / 2
    centres = [cx0 + i * (cw + cgap) + cw / 2 for i in range(3)]
    b.append(path(f'M{npz_cx:.1f} {ys+bh}L{npz_cx:.1f} {bus_y}L{centres[0]:.1f} {bus_y}',
                  T['wire'], 1.6))
    cards = [('teleop.py', 'drive the trained policy from the keyboard'),
             ('sim_viewer', 'three.js replay — live in Colab, standalone anywhere'),
             ('NumpyPolicy', '~40 lines of NumPy · matches SB3 to 6e-07')]
    cy, ch = 200, 64
    for cxc, (t, s) in zip(centres, cards):
        x = cxc - cw / 2
        b.append(arrow(cxc, bus_y, cxc, cy - 6, T['wire'], 1.6))
        b.append(rr(x, cy, cw, ch, 12, T['card'], T['border'], 1))
        b.append(circ(x + 20, cy + 26, 3.5, T['green']))
        b.append(tx(x + 32, cy + 30, t, 13, T['text'], weight=700, mono=True))
        b.append(tx(x + 20, cy + 50, s, 10.5, T['muted']))
    return W, H, ''.join(b), 'QuadBot pipeline: RobotSpec to MJCF to Gymnasium env to PPO to an exported NumPy policy'


def d_control_loop(T):
    W, H = 960, 348
    b = [tx(W / 2, 30, 'ONE CONTROL STEP — 20 ms', 11.5, T['muted'], 'middle', 700, ls=1.8)]
    top = [('actor  π(o)', '48 → 256 → 256 → 128 → 12', 'ELU, deterministic at eval', T['accent']),
           ('action  a ∈ [−1,1]¹²', 'q* = q_default + 0.3 · a', 'joint-position offsets', T['yellow']),
           ('PD servos', 'kp 50   kd 1.5', 'τ clipped at 25 / 25 / 35 N·m', T['cyan']),
           ('MuJoCo', '5 × 4 ms = 250 Hz', 'contacts, friction, gravity', T['purple'])]
    bw, gap, ys, bh = 208, 32, 52, 86
    x0 = (W - (4 * bw + 3 * gap)) / 2
    cxs = []
    for i, (t, s1, s2, c) in enumerate(top):
        x = x0 + i * (bw + gap)
        cxs.append(x + bw / 2)
        b.append(rr(x, ys, bw, bh, 12, T['card'], T['border'], 1))
        b.append(rr(x + 16, ys + 14, 24, 4, 2, c))
        b.append(tx(x + 16, ys + 40, t, 13.5, T['text'], weight=700))
        b.append(tx(x + 16, ys + 58, s1, 10.5, c, mono=True))
        b.append(tx(x + 16, ys + 74, s2, 10.5, T['muted']))
        if i < 3:
            b.append(arrow(x + bw + 5, ys + bh / 2, x + bw + gap - 5, ys + bh / 2, T['wire'], 1.6))
    # return path
    ry, rh = 190, 72
    rw = 300
    ex, gx = x0, x0 + 4 * bw + 3 * gap - rw
    for x, t, s1, c in ((gx, 'state', 'qpos · qvel · foot contacts', T['metal']),
                        (ex, 'observation (48)', 'body frame, scaled — the loop closes', T['green'])):
        b.append(rr(x, ry, rw, rh, 12, T['card'], T['border'], 1))
        b.append(rr(x + 16, ry + 14, 24, 4, 2, c))
        b.append(tx(x + 16, ry + 40, t, 13.5, T['text'], weight=700))
        b.append(tx(x + 16, ry + 58, s1, 10.5, T['muted']))
    b.append(elbow([(cxs[3], ys + bh), (cxs[3], ry - 6)], T['wire'], 1.6))
    b.append(arrow(gx - 6, ry + rh / 2, ex + rw + 6, ry + rh / 2, T['wire'], 1.6))
    b.append(tx((gx + ex + rw) / 2, ry + rh / 2 - 10, '50 Hz policy rate', 10, T['dim'], 'middle', 600))
    b.append(elbow([(cxs[0], ry), (cxs[0], ys + bh + 6)], T['green'], 1.8))
    # reward branch
    bay = 292
    b.append(line(gx + rw / 2, ry + rh, gx + rw / 2, bay - 4, T['wire'], 1.4, dash='4 4'))
    b.append(rr(x0, bay, 4 * bw + 3 * gap, 38, 10, T['card2'], T['border'], 1, dash='5 4'))
    b.append(circ(x0 + 22, bay + 19, 3.5, T['accent']))
    b.append(tx(x0 + 34, bay + 23, 'reward', 12, T['text'], weight=700, mono=True))
    b.append(tx(x0 + 90, bay + 23, '14 shaped terms, clipped at 0 — consumed by PPO during training only',
                11, T['muted']))
    return W, H, ''.join(b), 'The QuadBot control loop: policy at 50 Hz driving PD servos over MuJoCo physics at 250 Hz'

def d_observation(T):
    W, H = 960, 330
    b = [tx(W / 2, 30, 'WHAT THE POLICY SEES — 48 NUMBERS, ALL IN THE BODY FRAME',
            11.5, T['muted'], 'middle', 700, ls=1.6)]
    segs = [('proj. gravity', 3, T['purple'], 0, '×1'),
            ('base lin vel', 3, T['cyan'], 0, '×2.0'),
            ('base ang vel', 3, T['cyan'], 1, '×0.25'),
            ('command', 3, T['accent'], 0, '×2, 2, 0.25'),
            ('joint positions', 12, T['green'], 0, '− q default'),
            ('joint velocities', 12, T['green'], 1, '×0.05'),
            ('previous action', 12, T['yellow'], 0, '×1')]
    b.append('<defs><pattern id="qbhatch" width="7" height="7" patternUnits="userSpaceOnUse"'
             ' patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="7" stroke="%s"'
             ' stroke-width="3" opacity="0.4"/></pattern></defs>' % T['bg'])
    x0, bw, by, bh = 40, 880, 96, 48
    cell = bw / 48.0
    tiers = [50, 74, 50, 74]
    idx = 0
    for i, (name, n, col, hatch, scale) in enumerate(segs):
        x, w = x0 + idx * cell, n * cell
        b.append(rr(x + 1, by, w - 2, bh, 4, col, 'none'))
        if hatch:
            b.append(rr(x + 1, by, w - 2, bh, 4, 'url(#qbhatch)', 'none'))
        for k in range(1, n):
            b.append(line(x + k * cell, by + 5, x + k * cell, by + bh - 5, T['bg'], 1, op=0.45))
        if n == 12:
            b.append(tx(x + w / 2, by + bh / 2 - 1, name, 11.5, T['bg'], 'middle', 700))
            b.append(tx(x + w / 2, by + bh / 2 + 15, scale, 9.5, T['bg'], 'middle', mono=True, op=0.92))
        else:
            ly = tiers[i]
            b.append(line(x + w / 2, ly + 5, x + w / 2, by - 3, col, 1.2, op=0.7))
            b.append(tx(x + w / 2, ly, name, 10.5, T['text'], 'middle', 600))
            b.append(tx(x + w / 2, ly - 13, scale, 9.5, col, 'middle', mono=True))
        idx += n
    for k in (0, 3, 6, 9, 12, 24, 36, 48):
        b.append(tx(x0 + k * cell, by + bh + 17, str(k), 9.5, T['dim'],
                    'start' if k == 0 else ('end' if k == 48 else 'middle'), mono=True))

    # ---- action, drawn under the observation slice it feeds back into ----
    xa, ay, ah = x0 + 36 * cell, 212, 46
    b.append(tx(x0, 202, 'ACTION — 12 JOINT-POSITION OFFSETS', 10, T['muted'], weight=700, ls=1.3))
    b.append(tx(x0, 230, 'a ∈ [−1, 1]¹²    →    q* = q default + 0.3 · a   [rad]',
                12.5, T['text'], weight=600, mono=True))
    b.append(tx(x0, 252, "the policy's own output comes back as the last twelve numbers "
                         'of the next observation', 10.5, T['muted']))
    for leg_i, leg in enumerate(('FL', 'FR', 'RL', 'RR')):
        gx = xa + leg_i * 3 * cell
        b.append(rr(gx + 1, ay, 3 * cell - 2, ah, 5, T['card2'], T['yellow'], 1.2))
        b.append(tx(gx + 1.5 * cell, ay + 29, leg, 13, T['text'], 'middle', 700, mono=True))
    b.append(tx(xa + 6 * cell, ay + ah + 16, 'abad · hip · knee, for each leg',
                9.5, T['dim'], 'middle'))
    fx = xa + 6 * cell
    b.append(arrow(fx, ay - 6, fx, by + bh + 28, T['yellow'], 1.4, dash='4 3', s=6))
    b.append(tx(fx + 11, ay - 26, 'fed back', 9.5, T['yellow'], weight=600))

    chips = [('policy rate', '50 Hz', T['cyan']), ('physics', '250 Hz', T['purple']),
             ('episode', '1000 steps = 20 s', T['green']),
             ('command resample', 'every 10 s', T['accent'])]
    cx = x0
    for lab, val, col in chips:
        s, w = chip(cx, 288, lab, val, T, col, h=26)
        b.append(s)
        cx += w + 10
    return (W, H, ''.join(b),
            'The 48-dimensional observation vector and the 12-dimensional action, segment by segment')


def d_reward(T):
    W, H = 960, 302
    b = [tx(W / 2, 30, 'FOURTEEN REWARD TERMS — AND NOT ONE OF THEM MENTIONS A GAIT',
            11.5, T['muted'], 'middle', 700, ls=1.6)]
    groups = [('track the command', T['cyan'],
               [('tracking_lin_vel', '+1.5'), ('tracking_ang_vel', '+0.5')]),
              ('stay level and tall', T['purple'],
               [('base_height', '−30.0'), ('orientation', '−5.0'),
                ('lin_vel_z', '−2.0'), ('ang_vel_xy', '−0.05')]),
              ('step cleanly', T['green'],
               [('feet_air_time', '+1.0'), ('collision', '−1.0'), ('feet_slip', '−0.1')]),
              ('move smoothly', T['yellow'],
               [('action_rate', '−0.01'), ('torques', '−1e−4'),
                ('dof_acc', '−2.5e−7')]),
              ('respect the hardware', T['accent'],
               [('dof_pos_limits', '−10.0'), ('abad_deviation', '−0.5')])]
    cw, gap, cy, ch = 172, 14, 56, 148
    x0 = (W - (5 * cw + 4 * gap)) / 2
    for i, (title, col, terms) in enumerate(groups):
        x = x0 + i * (cw + gap)
        b.append(rr(x, cy, cw, ch, 12, T['card'], T['border'], 1))
        b.append(rr(x + 12, cy + 14, 22, 4, 2, col))
        b.append(tx(x + 12, cy + 38, title, 11.5, T['text'], weight=700))
        b.append(line(x + 12, cy + 50, x + cw - 12, cy + 50, T['border'], 1))
        for j, (name, wgt) in enumerate(terms):
            yy = cy + 70 + j * 24
            pos = wgt.startswith('+')
            b.append(tx(x + 12, yy, name, 9.5, T['text'] if pos else T['muted'], mono=True))
            b.append(tx(x + cw - 12, yy, wgt, 10, T['green'] if pos else T['red'], 'end', 700, mono=True))
    notes = [('tracking', 'exp(−e² / 0.25)', T['cyan']),
             ('total', 'clipped at 0', T['green']),
             ('gait terms', 'none', T['accent']),
             ('tuned after', 'legged_gym', T['purple'])]
    cx = x0
    for lab, val, col in notes:
        s, w = chip(cx, 230, lab, val, T, col, h=26)
        b.append(s)
        cx += w + 10
    b.append(tx(W - x0, 272, 'the trot is what PPO found on its own', 11, T['dim'], 'end', 600))
    return (W, H, ''.join(b),
            'The fourteen shaped reward terms, grouped by what they ask the robot to do')



def _dim_v(x, y1, y2, label, T, side=-1, off=8):
    s = line(x, y1, x, y2, T['dim'], 1)
    s += line(x - 4, y1, x + 4, y1, T['dim'], 1) + line(x - 4, y2, x + 4, y2, T['dim'], 1)
    s += tx(x + side * off, (y1 + y2) / 2 + 3.5, label, 10, T['muted'],
            'end' if side < 0 else 'start', 600, mono=True)
    return s


def _dim_h(y, x1, x2, label, T, above=True):
    s = line(x1, y, x2, y, T['dim'], 1)
    s += line(x1, y - 4, x1, y + 4, T['dim'], 1) + line(x2, y - 4, x2, y + 4, T['dim'], 1)
    s += tx((x1 + x2) / 2, y - 7 if above else y + 14, label, 10, T['muted'], 'middle', 600, mono=True)
    return s


def d_robot(T):
    W, H = 960, 458
    b = [tx(W / 2, 30, 'THE ROBOT IS GENERATED, NOT DOWNLOADED - EVERY NUMBER IS A FIELD ON RobotSpec',
            11.5, T['muted'], 'middle', 700, ls=1.4)]
    for px in (20, 490):
        b.append(rr(px, 48, 450, 340, 12, T['card'], T['border'], 1))
    b.append(tx(40, 72, 'SIDE VIEW · NOMINAL STANCE', 10, T['muted'], weight=700, ls=1.3))
    b.append(tx(510, 72, 'TOP VIEW · LEG NAMING', 10, T['muted'], weight=700, ls=1.3))

    # ---------------- side view ----------------
    S = 400.0
    gy = 344.0
    cx = 250.0
    cy = gy - 0.3143 * S
    b.append(line(44, gy, 452, gy, T['wire'], 1.6))
    for gx in range(48, 452, 18):
        b.append(line(gx, gy, gx - 7, gy + 7, T['wire'], 1, op=0.5))

    def leg(hx, col, op):
        tt = (hx - 0.1435 * S, cy + 0.1393 * S)
        ct = (tt[0] + 0.1288 * S, tt[1] + 0.1530 * S)
        s = line(hx, cy, tt[0], tt[1], col, 9, cap='round', op=op)
        s += line(tt[0], tt[1], ct[0], ct[1], col, 8, cap='round', op=op)
        s += circ(ct[0], ct[1], 0.022 * S, col, op=op)
        return s, tt, ct

    rear, rtt, rct = leg(cx - 0.18 * S, T['wire'], 0.85)
    b.append(rear)
    b.append(rr(cx - 0.19 * S, cy - 0.045 * S, 0.38 * S, 0.09 * S, 7, T['metal'], T['border'], 1))
    b.append(rr(cx - 0.19 * S + 6, cy - 0.045 * S + 5, 0.38 * S - 12, 7, 3.5, T['accent'], op=0.9))
    front, ftt, fct = leg(cx + 0.18 * S, T['metal'], 1.0)
    b.append(front)
    hx = cx + 0.18 * S
    for jx, jy in ((hx, cy), ftt):
        b.append(circ(jx, jy, 5.5, T['bg'], T['accent'], 2.4))
    b.append(circ(cx - 0.18 * S, cy, 4.5, T['bg'], T['wire'], 2))

    b.append(_dim_v(74, cy, gy, '0.31 m', T, -1, 6))
    b.append(tx(68, (cy + gy) / 2 + 17, 'stance', 9.5, T['dim'], 'end'))
    b.append(_dim_h(cy - 0.045 * S - 16, cx - 0.19 * S, cx + 0.19 * S, '0.38 m', T))
    ann = [(((hx + ftt[0]) / 2, (cy + ftt[1]) / 2), 'thigh 0.20 m'),
           ((((ftt[0] + fct[0]) / 2), (ftt[1] + fct[1]) / 2), 'calf 0.20 m'),
           (fct, 'foot ⌀ 0.044 m')]
    for k, (pt, lab) in enumerate(ann):
        ax, ay = pt
        lx, ly = 372, 232 + k * 34
        b.append(line(ax + 6, ay, lx - 6, ly - 3.5, T['dim'], 1, dash='3 3'))
        b.append(tx(lx, ly, lab, 10, T['text'], weight=600))
    b.append(tx(hx + 12, cy - 12, 'abad ±0.80', 9.5, T['accent'], weight=600, mono=True))
    b.append(tx(ftt[0] - 12, ftt[1] - 17, 'knee', 9.5, T['accent'], 'end', 600, mono=True))
    b.append(tx(ftt[0] - 12, ftt[1] - 5, '−2.70…−0.70', 9, T['dim'], 'end', mono=True))
    b.append(tx(hx + 12, cy + 4, 'hip −1.0…2.6', 9.5, T['dim'], mono=True))

    # ---------------- top view ----------------
    tcx, tcy = 712.0, 208.0
    hl, hw = 0.38 * S / 2, 0.18 * S / 2
    b.append(rr(tcx - hw, tcy - hl, 2 * hw, 2 * hl, 9, T['metal'], T['border'], 1))
    b.append(rr(tcx - 22, tcy - hl + 12, 44, 2 * hl - 24, 6, T['accent'], op=0.9))
    b.append(tx(566, tcy - 4, 'torso', 11, T['text'], weight=700))
    b.append(tx(566, tcy + 13, '0.38 × 0.18 × 0.09 m', 9.5, T['muted'], mono=True))
    b.append(tx(566, tcy + 28, '5.5 kg', 9.5, T['muted'], mono=True))
    for name, sx, sy in (('FL', -1, -1), ('FR', 1, -1), ('RL', -1, 1), ('RR', 1, 1)):
        ax_ = tcx + sx * 0.055 * S
        ay_ = tcy + sy * 0.18 * S
        ex = ax_ + sx * 0.07 * S
        col = T['cyan'] if name in ('FL', 'RR') else T['accent']
        b.append(line(ax_, ay_, ex, ay_, T['metal'], 7, cap='round'))
        b.append(circ(ax_, ay_, 4.5, T['bg'], col, 2.2))
        b.append(circ(ex, ay_ + sy * 6, 8.8, col, op=0.9))
        b.append(tx(ex + sx * 20, ay_ + sy * 6 + 4, name, 11.5, col,
                    'start' if sx > 0 else 'end', 700, mono=True))
    b.append(_dim_v(tcx + 0.125 * S + 54, tcy - 0.18 * S, tcy + 0.18 * S, '0.36 m', T, 1, 7))
    b.append(_dim_h(tcy + 0.18 * S + 44, tcx - 0.055 * S, tcx + 0.055 * S, '0.11 m', T, False))
    b.append(_dim_h(tcy - 0.18 * S - 32, tcx - 0.125 * S, tcx - 0.055 * S, '0.07 m', T))
    ox, oy = 560, 336
    b.append(arrow(ox, oy, ox, oy - 34, T['dim'], 1.3, s=5.5))
    b.append(arrow(ox, oy, ox - 34, oy, T['dim'], 1.3, s=5.5))
    b.append(tx(ox + 5, oy - 30, 'x fwd', 9, T['dim'], mono=True))
    b.append(tx(ox - 36, oy + 13, 'y left', 9, T['dim'], mono=True))
    b.append(tx(ox - 6, oy + 32, 'diagonal pairs share a colour', 9, T['dim'], op=0.9))

    chips = [('total mass', '11.3 kg', T['green']), ('actuated joints', '12', T['cyan']),
             ('servo gains', 'kp 50 · kd 1.5', T['purple']),
             ('torque limits', '25 / 25 / 35 N·m', T['accent']),
             ('foot friction', '1.0', T['yellow'])]
    cx2 = 20
    for lab, val, col in chips:
        s, w = chip(cx2, 406, lab, val, T, col, h=26)
        b.append(s)
        cx2 += w + 9
    return (W, H, ''.join(b),
            'QuadBot geometry: side view of the nominal stance and top view of the four legs')


def d_network(T):
    W, H = 960, 306
    b = [tx(W / 2, 30, 'ACTOR AND CRITIC - TWO SEPARATE 256 · 256 · 128 TRUNKS',
            11.5, T['muted'], 'middle', 700, ls=1.6)]
    ix, iy, iw, ih = 24, 124, 130, 60
    b.append(rr(ix, iy, iw, ih, 11, T['card'], T['border'], 1))
    b.append(tx(ix + iw / 2, iy + 26, 'observation', 12, T['text'], 'middle', 700))
    b.append(tx(ix + iw / 2, iy + 46, '48', 14, T['green'], 'middle', 700, mono=True))
    lw, lh, gap, lx0 = 104, 60, 26, 190
    rows = [(68, T['accent'], 'actor  π(o)', 'μ  (12)', 'plus a state-independent log σ'),
            (188, T['cyan'], 'critic  V(o)', 'V  (1)', 'dropped after training')]
    for ry, col, name, headlab, sub in rows:
        b.append(arrow(ix + iw + 4, iy + ih / 2, lx0 - 8, ry + lh / 2, col, 1.6))
        b.append(tx(lx0, ry - 10, name, 11, col, weight=700, mono=True))
        for j, n in enumerate(('256', '256', '128')):
            x = lx0 + j * (lw + gap)
            b.append(rr(x, ry, lw, lh, 10, T['card'], col, 1.4))
            b.append(tx(x + lw / 2, ry + 28, n, 15, T['text'], 'middle', 700, mono=True))
            b.append(tx(x + lw / 2, ry + 46, 'ELU', 9.5, T['muted'], 'middle'))
            nx = x + lw
            b.append(arrow(nx + 5, ry + lh / 2, nx + gap - 5, ry + lh / 2, T['wire'], 1.5, s=6))
        hx = lx0 + 3 * (lw + gap)
        b.append(rr(hx, ry, 150, lh, 10, col, 'none', op=0.92))
        b.append(tx(hx + 75, ry + 27, headlab, 14, T['bg'], 'middle', 700, mono=True))
        b.append(tx(hx + 75, ry + 45, sub, 8.6, T['bg'], 'middle', op=0.85))
    ex, ey = 790, 96
    b.append(arrow(lx0 + 3 * (lw + gap) + 154, 98, ex - 6, ey + 24, T['green'], 1.6, dash='4 3'))
    b.append(rr(ex, ey, 146, 100, 11, T['card'], T['green'], 1.4))
    b.append(tx(ex + 73, ey + 26, 'policy.npz', 12, T['green'], 'middle', 700, mono=True))
    b.append(tx(ex + 73, ey + 48, '4 weights + 4 biases', 9.5, T['muted'], 'middle'))
    b.append(tx(ex + 73, ey + 66, '444 KB', 11, T['text'], 'middle', 600, mono=True))
    b.append(tx(ex + 73, ey + 84, 'runs on NumPy alone', 9.5, T['muted'], 'middle'))
    chips = [('optimiser', 'PPO', T['accent']),
             ('normalisation', 'VecNormalize obs + reward', T['purple']),
             ('log_std_init', '−0.5', T['cyan']),
             ('SB3 vs NumPy', 'max |Δ| 6e−07', T['green'])]
    cx = 24
    for lab, val, col in chips:
        s, w = chip(cx, 268, lab, val, T, col, h=26)
        b.append(s)
        cx += w + 10
    return (W, H, ''.join(b),
            'Network architecture: separate actor and critic trunks, and the exported NumPy actor')


def d_gait(T):
    W, H = 960, 344
    b = [tx(W / 2, 30, 'THE EMERGENT TROT - DIAGONAL PAIRS SHARE A COLOUR',
            11.5, T['muted'], 'middle', 700, ls=1.6)]
    x0, x1 = 96.0, 830.0
    span = 2.0
    TT = 1.0 / 1.5
    px = (x1 - x0) / span
    rows = [('FL', 0.0, 0.52, T['cyan']), ('FR', 0.5, 0.4875, T['accent']),
            ('RL', 0.5, 0.5025, T['accent']), ('RR', 0.0, 0.5525, T['cyan'])]
    top, pitch, bh = 88, 42, 28
    b.append(rr(x0, top - 10, x1 - x0, 4 * pitch + 2, 8, T['card'], T['border'], 1))
    for k in range(1, 4):
        gx = x0 + k * 0.5 * px
        b.append(line(gx, top - 10, gx, top - 8 + 4 * pitch, T['border'], 1, op=0.9))
    for i, (name, ph, duty, col) in enumerate(rows):
        y = top + i * pitch
        b.append(tx(x0 - 16, y + bh / 2 + 4, name, 12, col, 'end', 700, mono=True))
        for k in range(-1, 4):
            a = max(0.0, (ph + k) * TT)
            c = min(span, (ph + k) * TT + duty * TT)
            if (c - a) * px > 2:
                b.append(rr(x0 + a * px, y, (c - a) * px, bh, 5, col, 'none', op=0.92))
        b.append(tx(x1 + 16, y + bh / 2 + 4, '%.2f' % duty, 10.5, T['muted'], 'start', 600, mono=True))
    b.append(tx(x1 + 16, top - 16, 'duty', 9, T['dim'], 'start', 700))
    ay = top + 4 * pitch + 4
    b.append(line(x0, ay, x1, ay, T['wire'], 1.2))
    for t in (0.0, 0.5, 1.0, 1.5, 2.0):
        gx = x0 + t * px
        b.append(line(gx, ay, gx, ay + 5, T['wire'], 1.2))
        b.append(tx(gx, ay + 18, '%.1f' % t, 9.5, T['dim'], 'middle', mono=True))
    b.append(tx(x1 + 16, ay + 18, 'time [s]', 9.5, T['dim'], 'start', mono=True))
    bx1, bx2, byy = x0, x0 + TT * px, top - 24
    b.append(path('M%.1f %.1fL%.1f %.1fL%.1f %.1fL%.1f %.1f'
                  % (bx1, byy + 6, bx1, byy, bx2, byy, bx2, byy + 6), T['dim'], 1.2))
    b.append(tx((bx1 + bx2) / 2, byy - 5, 'T = 0.67 s  ·  1.5 strides/s',
                9.5, T['muted'], 'middle', 600))
    chips = [('diagonal feet in phase', '92%', T['cyan']),
             ('diagonal correlation', '+0.84', T['cyan']),
             ('lateral correlation', '−0.84', T['accent']),
             ('duty factor', '≈0.50', T['green'])]
    cx = 40
    for lab, val, col in chips:
        s, w = chip(cx, 290, lab, val, T, col, h=26)
        b.append(s)
        cx += w + 10
    b.append(tx(W / 2, 334,
                'phasing reconstructed from the measured duty factors and contact correlations '
                '— the raw trace is runs/main/footfall.png', 9, T['dim'], 'middle'))
    return (W, H, ''.join(b),
            'Footfall diagram of the emergent trot: diagonal pairs FL+RR and FR+RL alternate')


DIAGRAMS = [('pipeline', d_pipeline), ('control-loop', d_control_loop),
            ('observation', d_observation), ('robot', d_robot),
            ('reward', d_reward), ('network', d_network), ('gait', d_gait)]


def main():
    import io as _io
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, 'assets', 'diagrams')
    os.makedirs(out, exist_ok=True)
    for name, fn in DIAGRAMS:
        for T in THEMES:
            w, h, body, title = fn(T)
            p = os.path.join(out, '%s-%s.svg' % (name, T['name']))
            with _io.open(p, 'w', encoding='utf-8') as f:
                f.write(svg(w, h, T, body, title))
            print(os.path.relpath(p, here))


if __name__ == '__main__':
    main()
