"""
arena_build.py - construye arena.xml para una pista dada del catalogo.

  ensure_assets()                 crea textures/ y meshes/ si faltan
  build_arena(track, n_slots=12, path="arena.xml")
                                  arma el XML (mismo lab, la pista cambia) con
                                  un pool de n_slots obstaculos (free joint).

El laboratorio/fondo es identico para todas las pistas; solo cambian los geoms
de asfalto/linea (track.road_geoms) y el punto de inicio del robot.
"""
import os
import numpy as np
from PIL import Image

ROAD_RGBA = "0.22 0.22 0.25 1"
LINE_RGBA = "0.95 0.85 0.05 1"
N_TEX = 8


# ----------------------------- assets -----------------------------
def ensure_assets():
    if os.path.isdir("textures") and os.path.isdir("meshes") \
       and len(os.listdir("textures")) >= N_TEX:
        return
    os.makedirs("textures", exist_ok=True)
    os.makedirs("meshes", exist_ok=True)
    rng = np.random.default_rng(7)

    def smooth(n, s):
        small = rng.random((s, s, 3))
        return np.array(Image.fromarray((small*255).astype(np.uint8)).resize((n, n), Image.BICUBIC))
    makers = []
    makers.append(lambda: smooth(256, 16))
    makers.append(lambda: (rng.random((256, 256, 3))*255).astype(np.uint8))

    def stripes():
        c1, c2 = rng.random(3), rng.random(3); p = rng.integers(8, 28)
        img = np.zeros((256, 256, 3))
        for i in range(256):
            img[i, :] = c1 if (i//p) % 2 == 0 else c2
        return (img*255).astype(np.uint8)

    def checker():
        c1, c2 = rng.random(3), rng.random(3); p = rng.integers(12, 40)
        xs = (np.arange(256)//p)[:, None]; ys = (np.arange(256)//p)[None, :]
        return (np.where(((xs+ys) % 2).astype(bool)[..., None], c1, c2)*255).astype(np.uint8)

    def dots():
        bg, fg = rng.random(3), rng.random(3); img = np.ones((256, 256, 3))*bg
        for _ in range(rng.integers(20, 60)):
            cy, cx, r = rng.integers(0, 256), rng.integers(0, 256), rng.integers(6, 20)
            yy, xx = np.ogrid[:256, :256]; img[(yy-cy)**2+(xx-cx)**2 < r*r] = fg
        return (img*255).astype(np.uint8)

    def marble():
        base = smooth(256, 8).mean(2); v = (np.sin(base/255*6*np.pi+base/40)+1)/2
        c1, c2 = rng.random(3), rng.random(3)
        return (c1[None, None]*v[..., None]+c2[None, None]*(1-v[..., None])*255).astype(np.uint8)
    makers += [stripes, checker, dots, marble, stripes, dots]
    for i, mk in enumerate(makers[:N_TEX]):
        Image.fromarray(mk()).save(f"textures/tex{i}.png")
    with open("meshes/pyramid.obj", "w") as f:
        f.write("v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\nv 0 0 1\n"
                "f 1 2 5\nf 2 3 5\nf 3 4 5\nf 4 1 5\nf 1 4 3\nf 1 3 2\n")
    with open("meshes/prism.obj", "w") as f:
        f.write("v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\n"
                "v -0.5 0 0.9\nv 0.5 0 0.9\n"
                "f 1 2 6\nf 1 6 5\nf 4 3 6\nf 4 6 5\nf 1 4 5\nf 2 3 6\nf 1 2 3\nf 1 3 4\n")


# ----------------------------- lab (igual para todas las pistas) -----------------------------
def _desk(cx, cy, ang, top):
    g = [f'    <body pos="{cx} {cy} 0" euler="0 0 {ang}">',
         f'      <geom type="box" pos="0 0 0.74" size="0.6 0.3 0.02" rgba="{top}"/>']
    for sx in (-0.55, 0.55):
        for sy in (-0.25, 0.25):
            g.append(f'      <geom type="box" pos="{sx} {sy} 0.37" size="0.02 0.02 0.37" rgba="0.12 0.12 0.12 1"/>')
    g.append('      <geom type="box" pos="0.1 0 0.86" size="0.16 0.02 0.1" rgba="0.04 0.04 0.05 1"/>')
    g.append('    </body>'); return "\n".join(g)


def _person(cx, cy, ang, shirt, h=1.0):
    return "\n".join([
        f'    <body pos="{cx} {cy} 0" euler="0 0 {ang}">',
        f'      <geom type="box" pos="0 0 {h}" size="0.16 0.11 0.22" rgba="{shirt}"/>',
        f'      <geom type="sphere" pos="0 0 {h+0.32:.2f}" size="0.1" rgba="0.82 0.66 0.54 1"/>',
        f'      <geom type="box" pos="0.36 -0.08 0.4" size="0.07 0.06 0.4" rgba="0.1 0.1 0.12 1"/>',
        f'      <geom type="box" pos="0.36 0.08 0.4" size="0.07 0.06 0.4" rgba="0.1 0.1 0.12 1"/>',
        '    </body>'])


def _stool(cx, cy):
    return (f'    <body pos="{cx} {cy} 0">'
            f'<geom type="cylinder" pos="0 0 0.6" size="0.16 0.02" rgba="0.2 0.2 0.22 1"/>'
            f'<geom type="cylinder" pos="0 0 0.3" size="0.025 0.3" rgba="0.3 0.3 0.32 1"/></body>')


def _lab_clutter():
    tops = ["0.85 0.82 0.78 1", "0.6 0.45 0.3 1", "0.75 0.76 0.8 1"]
    shirts = ["0.1 0.1 0.12 1", "0.15 0.2 0.45 1", "0.5 0.12 0.12 1", "0.2 0.2 0.22 1",
              "0.1 0.3 0.25 1", "0.35 0.35 0.4 1", "0.4 0.3 0.1 1"]
    spots = ([(-3.2, yy, 0.0) for yy in (-2.4, -1.2, 0, 1.2, 2.4)] +
             [(3.2, yy, 3.1416) for yy in (-2.4, -1.2, 0, 1.2, 2.4)] +
             [(xx, 3.2, -1.5708) for xx in (-2, -0.7, 0.7, 2)] +
             [(xx, -3.2, 1.5708) for xx in (-2, -0.7, 0.7, 2)])
    out = []
    for k, (dx, dy, da) in enumerate(spots):
        out.append(_desk(dx, dy, da, tops[k % len(tops)]))
        if k % 2 == 0:
            offx = 0.55 if dx < -0.1 else (-0.55 if dx > 0.1 else 0)
            offy = -0.55 if dy > 0.1 else (0.55 if dy < -0.1 else 0)
            out.append(_person(dx+offx, dy+offy, da+3.1416, shirts[k % len(shirts)]))
        else:
            out.append(_stool(dx + (0.5 if dx < 0 else -0.5), dy))
    out.append(_person(0, -2.6, 1.5708, shirts[2], h=1.05))
    out.append(_person(-2.5, 0, 0, shirts[1], h=1.05))
    return "\n".join(out)


# ----------------------------- obstaculos + distractores -----------------------------
def _obstacle_pool(n_slots, rng):
    shapes = ["box", "sphere", "cylinder", "ellipsoid", "capsule", "mesh:pyramid",
              "box", "sphere", "cylinder", "mesh:prism", "ellipsoid", "capsule"]
    out = []
    for i in range(n_slots):
        sh = shapes[i % len(shapes)]; mat = f'material="m{i % N_TEX}"'
        s = rng.uniform(0.022, 0.045)
        if sh.startswith("mesh:"):
            geom = f'<geom name="obs_{i}" type="mesh" mesh="{sh.split(":")[1]}" {mat}/>'
        elif sh == "box":
            geom = f'<geom name="obs_{i}" type="box" size="{s:.3f} {s:.3f} {s:.3f}" {mat}/>'
        elif sh == "sphere":
            geom = f'<geom name="obs_{i}" type="sphere" size="{s:.3f}" {mat}/>'
        elif sh == "cylinder":
            geom = f'<geom name="obs_{i}" type="cylinder" size="{s:.3f} {s*1.5:.3f}" {mat}/>'
        elif sh == "ellipsoid":
            geom = f'<geom name="obs_{i}" type="ellipsoid" size="{s:.3f} {s*0.7:.3f} {s*1.4:.3f}" {mat}/>'
        else:
            geom = f'<geom name="obs_{i}" type="capsule" size="{s*0.7:.3f} {s:.3f}" {mat}/>'
        out.append(f'    <body name="obs_{i}" pos="0 0 -5">\n      <freejoint/>\n      {geom}\n    </body>')
    return "\n".join(out)


def _dynamic_distractors():
    def dp(name, x, y, shirt):
        return (f'    <body name="{name}" mocap="true" pos="{x} {y} 0">'
                f'<geom type="box" pos="0 0 0.92" size="0.16 0.11 0.28" rgba="{shirt}" contype="0" conaffinity="0"/>'
                f'<geom type="sphere" pos="0 0 1.32" size="0.11" rgba="0.82 0.66 0.54 1" contype="0" conaffinity="0"/>'
                f'<geom type="box" pos="0 -0.09 0.32" size="0.07 0.06 0.32" rgba="0.1 0.1 0.12 1" contype="0" conaffinity="0"/>'
                f'<geom type="box" pos="0 0.09 0.32" size="0.07 0.06 0.32" rgba="0.1 0.1 0.12 1" contype="0" conaffinity="0"/></body>')

    def dt(name, x, y):
        return (f'    <body name="{name}" mocap="true" pos="{x} {y} 0">'
                f'<geom type="box" pos="0 0 0.74" size="0.6 0.3 0.02" rgba="0.7 0.55 0.35 1" contype="0" conaffinity="0"/>'
                f'<geom type="box" pos="0 0 0.37" size="0.05 0.05 0.37" rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/></body>')
    return "\n".join([dp("dyn_p0", -2.4, 2.2, "0.15 0.2 0.45 1"),
                      dp("dyn_p1", 2.4, -2.2, "0.5 0.12 0.12 1"),
                      dp("dyn_p2", -2.2, -2.4, "0.1 0.3 0.25 1"),
                      dt("dyn_t0", 2.6, 1.2), dt("dyn_t1", -2.8, -1.2)])


# ----------------------------- robot -----------------------------
def _robot(x0, y0, yaw0):
    return f"""    <body name="base" pos="{x0:.3f} {y0:.3f} 0.028" euler="0 0 {yaw0:.3f}">
      <freejoint name="root"/>
      <geom name="shell" type="ellipsoid" size="0.10 0.10 0.045" pos="0 0 0.037" rgba="1.0 0.93 0.55 1" mass="0.9"/>
      <geom name="nose" type="box" size="0.022 0.012 0.008" pos="0.092 0 0.03" rgba="0.95 0.45 0.1 1" mass="0.001"/>
      <site name="imu" pos="0 0 0.04" size="0.004"/>
      <body name="cam_mast" pos="0.06 0 0.12">
        <joint name="cam_pan" class="camlink" type="hinge" axis="0 0 1" range="-1.2 1.2"/>
        <geom class="camlink" type="cylinder" size="0.006 0.015" mass="0.01"/>
        <body name="cam_tilt" pos="0 0 0.015">
          <joint name="cam_tilt" class="camlink" type="hinge" axis="0 1 0" range="-0.5 1.0"/>
          <geom class="camlink" type="box" size="0.012 0.02 0.01" mass="0.01"/>
          <camera name="eye" pos="0.02 0 0" fovy="90" xyaxes="0 -1 0  0.7071 0 0.7071"/>
        </body>
      </body>
      <body name="left_wheel" pos="-0.01 0.085 0"><joint name="left_wheel" class="wheel"/><geom class="wheel"/></body>
      <body name="right_wheel" pos="-0.01 -0.085 0"><joint name="right_wheel" class="wheel"/><geom class="wheel"/></body>
      <geom name="front_caster" class="caster" pos="0.075 0 -0.018"/>
      <geom name="rear_caster"  class="caster" pos="-0.075 0 -0.018"/>
    </body>"""


# ----------------------------- ensamblaje -----------------------------
def build_arena(track, n_slots=12, path="arena.xml", seed=0):
    ensure_assets()
    rng = np.random.default_rng(seed)
    x0, y0 = track.start
    yaw0 = track.start_yaw
    road_xml, line_xml = track.road_geoms(ROAD_RGBA, LINE_RGBA)
    tex_assets = "\n".join(
        f'    <texture name="t{i}" type="2d" file="textures/tex{i}.png"/>\n'
        f'    <material name="m{i}" texture="t{i}" texrepeat="2 2"/>' for i in range(N_TEX))
    # marca inicio/fin perpendicular a la tangente
    px, py = -np.sin(yaw0), np.cos(yaw0)
    sx1, sy1 = x0+0.22*px, y0+0.22*py
    sx2, sy2 = x0-0.22*px, y0-0.22*py

    xml = f"""<mujoco model="arena">
  <compiler angle="radian" autolimits="true" meshdir="meshes"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <default>
    <default class="wheel"><geom type="cylinder" size="0.028 0.01" zaxis="0 1 0"
      rgba="0.12 0.12 0.12 1" friction="1.4 0.01 0.001" condim="3"/>
      <joint type="hinge" axis="0 1 0" damping="0.015" armature="0.003"/></default>
    <default class="caster"><geom type="sphere" size="0.008" rgba="0.4 0.4 0.4 1" condim="1"/></default>
    <default class="camlink"><geom contype="0" conaffinity="0" rgba="0.2 0.2 0.22 1"/>
      <joint damping="0.6"/></default>
  </default>
  <visual>
    <global offwidth="1280" offheight="1280"/>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.45 0.45 0.45"/>
    <quality shadowsize="0"/>
  </visual>
  <asset>
    <material name="floor" rgba="0.5 0.5 0.52 1" reflectance="0.05"/>
    <material name="wall" rgba="0.88 0.88 0.86 1"/>
    <mesh name="pyramid" file="pyramid.obj" scale="0.07 0.07 0.07"/>
    <mesh name="prism"   file="prism.obj"   scale="0.07 0.07 0.07"/>
{tex_assets}
  </asset>
  <worldbody>
    <light name="key"  pos="-2 -2 4" dir="0.5 0.5 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>
    <light name="fill" pos="2 2 4"   dir="-0.5 -0.5 -1" diffuse="0.5 0.5 0.5" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor" condim="3" friction="1.0 0.005 0.0001"/>
    <geom type="box" pos="-4.6 0 1.3" size="0.05 4.5 1.3" material="wall"/>
    <geom type="box" pos=" 4.6 0 1.3" size="0.05 4.5 1.3" material="wall"/>
    <geom type="box" pos="0 -4.5 1.3" size="4.6 0.05 1.3" material="wall"/>
    <geom type="box" pos="0  4.5 1.3" size="4.6 0.05 1.3" material="wall"/>
{_lab_clutter()}
    <geom name="start_l" type="box" pos="{sx1:.3f} {sy1:.3f} 0.05" size="0.015 0.015 0.05" rgba="0.1 0.9 0.2 1" contype="0" conaffinity="0"/>
    <geom name="start_r" type="box" pos="{sx2:.3f} {sy2:.3f} 0.05" size="0.015 0.015 0.05" rgba="0.1 0.9 0.2 1" contype="0" conaffinity="0"/>
{road_xml}
{line_xml}
{_obstacle_pool(n_slots, rng)}
{_dynamic_distractors()}
{_robot(x0, y0, yaw0)}
  </worldbody>
  <contact>
    <exclude body1="base" body2="left_wheel"/><exclude body1="base" body2="right_wheel"/>
  </contact>
  <actuator>
    <velocity name="left_motor"  joint="left_wheel"  kv="5.0" ctrlrange="-22 22"/>
    <velocity name="right_motor" joint="right_wheel" kv="5.0" ctrlrange="-22 22"/>
    <position name="pan_servo"  joint="cam_pan"  kp="6" ctrlrange="-1.2 1.2"/>
    <position name="tilt_servo" joint="cam_tilt" kp="6" ctrlrange="-0.5 1.0"/>
  </actuator>
  <sensor>
    <gyro name="gyro" site="imu"/>
    <framepos name="base_pos" objtype="site" objname="imu"/>
    <framequat name="base_quat" objtype="site" objname="imu"/>
  </sensor>
</mujoco>
"""
    with open(path, "w") as f:
        f.write(xml)
    return path
