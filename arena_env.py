"""
arena_env.py - logica del entorno (vision, obstaculos, iluminacion, episodios,
dinamismo y REWARD) sobre cualquier pista del catalogo (tracks.Track).
"""
from collections import defaultdict
import numpy as np
import mujoco

LINE_COLOR = (0.95, 0.85, 0.05)
ROAD_COLOR = (0.22, 0.22, 0.25)


# ============================== VISION ==============================
class Vision:
    def __init__(self, model, h=120, w=160, camera="eye", max_geom=20000):
        self.m, self.cam, self.h, self.w = model, camera, h, w
        self._max_geom = max(max_geom, model.ngeom + 2000)
        self._rgb = mujoco.Renderer(model, h, w, max_geom=self._max_geom)
        self._seg = self._make_seg()
        self._ngeom = model.ngeom
        self._last_ids = np.full((h, w), -1, np.int32)     # cache (fondo) -> nunca crashea
        self.line_ids = self._by_color(LINE_COLOR)
        self.road_ids = self._by_color(ROAD_COLOR)
        self.obs_ids = np.array([i for i in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("obs_")])

    def _by_color(self, c, tol=0.03):
        return np.array([i for i in range(self.m.ngeom)
                         if np.allclose(self.m.geom_rgba[i, :3], c, atol=tol)])

    def _make_seg(self):
        r = mujoco.Renderer(self.m, self.h, self.w, max_geom=self._max_geom)
        r.enable_segmentation_rendering()
        return r

    def rgb(self, d):
        self._rgb.update_scene(d, camera=self.cam); return self._rgb.render()

    def _ids(self, d):
        """Render de segmentacion ROBUSTO. Nunca crashea por IDs fuera de rango.
        - IDs invalidos -> fondo (-1).
        - Si el render falla, devuelve el ULTIMO mapa bueno (cache). NO recrea el
          renderer en cada fallo: recrearlo por frame desestabiliza el contexto GL
          en Windows y deja la mascara vacia -> el robot pierde la linea y gira."""
        try:
            self._seg.update_scene(d, camera=self.cam)
            ids = self._seg.render()[..., 0].astype(np.int32)
            ids[(ids >= self._ngeom) | (ids < -1)] = -1        # invalidos -> fondo
            self._last_ids = ids
            return ids
        except (IndexError, ValueError, RuntimeError):
            return self._last_ids                              # ultimo bueno -> sigue la linea

    def mask(self, d, kind="line"):
        g = self._ids(d)
        ids = {"line": self.line_ids, "road": self.road_ids, "obstacle": self.obs_ids}[kind]
        return np.isin(g, ids)

    def seg_map(self, d):
        g = self._ids(d)
        out = np.zeros((self.h, self.w, 3), np.uint8)
        out[np.isin(g, self.road_ids)] = (90, 90, 100)
        out[np.isin(g, self.line_ids)] = (240, 215, 15)
        out[np.isin(g, self.obs_ids)] = (235, 40, 40)
        return out

    def overlay(self, rgb, d):
        out = rgb.copy(); out[self.mask(d, "obstacle")] = (255, 40, 40); return out


# ============================== OBSTACULOS (slots + probabilidad) ==============================
class ObstacleField:
    def __init__(self, model, data, track):
        self.m, self.d, self.track = model, data, track
        self.qadr, self.dadr, self.gid, self.rbound, self.bid = [], [], [], [], []
        for i in range(999):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{i}")
            if bid < 0:
                break
            jid = model.body_jntadr[bid]
            self.bid.append(bid)
            self.qadr.append(int(model.jnt_qposadr[jid]))
            self.dadr.append(int(model.jnt_dofadr[jid]))
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}")
            self.gid.append(gid); self.rbound.append(float(model.geom_rbound[gid]))
        self.n = len(self.bid)                       # n_slots
        self.gid2k = {g: k for k, g in enumerate(self.gid)}
        self.active = []
        self._s = {}                                 # k -> arc-length sobre la pista

    def _park(self, k):
        a = self.qadr[k]
        self.d.qpos[a:a+3] = (0, 0, -5); self.d.qpos[a+3:a+7] = (1, 0, 0, 0)
        self.d.qvel[self.dadr[k]:self.dadr[k]+6] = 0

    def clear(self):
        for k in range(self.n):
            self._park(k)
        self.active = []; self._s = {}
        mujoco.mj_forward(self.m, self.d)

    def spawn(self, rng, prob=0.6, mode="probabilistic", avoid_xy=None, avoid_r=0.4):
        """slots: cada slot aparece con prob (probabilistic) o todos (fixed)."""
        for k in range(self.n):
            self._park(k)
        if mode == "fixed":
            chosen = list(range(self.n))
        else:
            chosen = [k for k in range(self.n) if rng.random() < prob]
        # puntos DISTINTOS de la pista, lejos del inicio
        idxs = [i for i in range(len(self.track.pts))
                if avoid_xy is None or
                np.hypot(*(self.track.pts[i]-avoid_xy)) >= avoid_r]
        rng.shuffle(idxs)
        self.active = []; self._s = {}
        for j, k in enumerate(chosen):
            if j >= len(idxs):
                break
            idx = idxs[j]
            px, py = self.track.pts[idx]
            ang = rng.uniform(0, 2*np.pi); off = rng.uniform(0.0, 0.22)
            px += off*np.cos(ang); py += off*np.sin(ang)
            a = self.qadr[k]
            self.d.qpos[a:a+3] = (px, py, self.rbound[k]+0.005)
            yaw = rng.uniform(0, 2*np.pi)
            self.d.qpos[a+3:a+7] = (np.cos(yaw/2), 0, 0, np.sin(yaw/2))
            self.d.qvel[self.dadr[k]:self.dadr[k]+6] = 0
            self.active.append(k); self._s[k] = float(self.track.cum[idx])
        mujoco.mj_forward(self.m, self.d)

    def obs_xy(self, k):
        a = self.qadr[k]; return self.d.qpos[a:a+2].copy()

    def obs_s(self, k):
        return self._s.get(k, 0.0)

    def tilt(self, k):
        up = self.d.xmat[self.bid[k]].reshape(3, 3)[:, 2]
        return float(np.arccos(np.clip(up[2], -1, 1)))


# ============================== ILUMINACION ==============================
class Lighting:
    def __init__(self, model):
        self.m = model
        self.base_diff = model.light_diffuse.copy()
        self.base_amb = np.array(model.vis.headlight.ambient).copy()
        self.base_hl = np.array(model.vis.headlight.diffuse).copy()

    def set(self, brightness):
        b = float(np.clip(brightness, 0.05, 2.0))
        self.m.light_diffuse[:] = np.clip(self.base_diff*b, 0, 1)
        self.m.vis.headlight.ambient[:] = np.clip(self.base_amb*b, 0, 1)
        self.m.vis.headlight.diffuse[:] = np.clip(self.base_hl*b, 0, 1)


# ============================== EPISODIO ==============================
class Episode:
    def __init__(self, model, data, track, seed=0):
        self.m, self.d, self.track = model, data, track
        self.rng = np.random.default_rng(seed)
        jid = model.joint("root").id
        self.qadr = int(model.jnt_qposadr[jid]); self.dadr = int(model.jnt_dofadr[jid])
        self.count = 0

    def _teleport(self, xy, yaw):
        a = self.qadr
        self.d.qpos[a:a+3] = (xy[0], xy[1], 0.028)
        self.d.qpos[a+3:a+7] = (np.cos(yaw/2), 0, 0, np.sin(yaw/2))
        self.d.qvel[self.dadr:self.dadr+6] = 0
        mujoco.mj_forward(self.m, self.d)

    def reset(self, field=None, obstacles=True, prob=0.6, mode="probabilistic",
              lighting=None, brightness=1.0, random_brightness=False, avoid_r=0.45):
        self._teleport(self.track.start, self.track.start_yaw)
        if field is not None:
            if obstacles:
                field.spawn(self.rng, prob, mode, avoid_xy=self.track.start, avoid_r=avoid_r)
            else:
                field.clear()
        if lighting is not None:
            b = self.rng.uniform(0.7, 1.2) if random_brightness else brightness
            lighting.set(b)
        self.count += 1


# ============================== DINAMISMO ==============================
class Dynamics:
    def __init__(self, model, data, p_person=0.18, p_table=0.05, seed=0):
        self.m, self.d = model, data
        self.p_person, self.p_table = p_person, p_table
        self.rng = np.random.default_rng(seed)
        self.people, self.tables = [], []
        for nm in ("dyn_p0", "dyn_p1", "dyn_p2"):
            mid = self._mocap(nm)
            if mid is not None:
                self.people.append({"mid": mid, "home": data.mocap_pos[mid].copy(),
                                    "walking": False, "target": None, "speed": 0.9})
        for nm in ("dyn_t0", "dyn_t1"):
            mid = self._mocap(nm)
            if mid is not None:
                self.tables.append({"mid": mid, "home": data.mocap_pos[mid].copy(), "target": None})

    def _mocap(self, name):
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        return None if bid < 0 else int(self.m.body_mocapid[bid])

    def _set_yaw(self, mid, yaw):
        self.d.mocap_quat[mid] = (np.cos(yaw/2), 0, 0, np.sin(yaw/2))

    def reset(self):
        for p in self.people:
            self.d.mocap_pos[p["mid"]] = p["home"]; p["walking"] = False
        for t in self.tables:
            self.d.mocap_pos[t["mid"]] = t["home"]; t["target"] = None

    def update(self, dt):
        for p in self.people:
            mid = p["mid"]
            if not p["walking"]:
                if self.rng.random() < self.p_person*dt:
                    side = self.rng.choice([-1, 1]); y0 = self.rng.uniform(-2.5, 2.5)
                    start = np.array([3.0*side, y0, 0.0])
                    end = np.array([-3.0*side, self.rng.uniform(-2.5, 2.5), 0.0])
                    self.d.mocap_pos[mid] = start; p["target"] = end; p["walking"] = True
                    self._set_yaw(mid, np.arctan2(end[1]-start[1], end[0]-start[0]))
            else:
                pos = self.d.mocap_pos[mid].copy(); vec = p["target"]-pos
                dd = np.linalg.norm(vec[:2])
                if dd < 0.05:
                    p["walking"] = False
                else:
                    pos[:2] += vec[:2]/dd*p["speed"]*dt; self.d.mocap_pos[mid] = pos
        for t in self.tables:
            mid = t["mid"]
            if t["target"] is None:
                if self.rng.random() < self.p_table*dt:
                    delta = self.rng.uniform(-0.3, 0.3, 3); delta[2] = 0
                    t["target"] = self.d.mocap_pos[mid]+delta
            else:
                pos = self.d.mocap_pos[mid].copy(); vec = t["target"]-pos
                dd = np.linalg.norm(vec[:2])
                if dd < 0.02:
                    t["target"] = None
                else:
                    pos[:2] += vec[:2]/dd*0.25*dt; self.d.mocap_pos[mid] = pos


# ============================== REWARD ==============================
class Rewards:
    """Reward sobre cualquier pista. Componentes (ver summary()):
      advance   + progress por metro AVANZADO en la direccion correcta
      safe      + chico por paso seguro (en pista, sin choque, derecho)
      arrival   + llegada correcta (+ bonus por rapidez)
      wrong_dir - llegada por direccion contraria (= -arrival en magnitud)
      timeout   - no llegar (base + por metro restante SOBRE LA PISTA)
      collision - choque (EVENTO con cooldown por objeto; no se acumula por frame)
      stuck     - estancado contra un objeto (acotado por segundo)
      avoid     + esquivar un obstaculo (una vez por obstaculo)
      offtrack  - fuera de la pista (por paso)
      ret       + volver a la pista (> costo de salirse)
      flip      - vuelco/inestabilidad (TERMINA)
    """
    DEFAULT_W = dict(progress=5.0, safe=0.05, arrival=20.0, speed_bonus=20.0,
                     wrong_dir=20.0, timeout=5.0, timeout_per_m=1.5,
                     collision=3.0, stuck=1.0, avoid=3.0, offtrack=0.10, ret=1.5, flip=8.0)

    def __init__(self, model, data, track, field, weights=None,
                 collision_cooldown=1.0, near_radius=0.45, far_lost=1.4,
                 flip_deg=60.0, stuck_after=0.5):
        self.m, self.d, self.track, self.field = model, data, track, field
        self.W = dict(self.DEFAULT_W)
        if weights:
            self.W.update(weights)
        self.collision_cooldown = collision_cooldown
        self.near_radius = near_radius
        self.far_lost = far_lost
        self.flip_thresh = np.radians(flip_deg)
        self.stuck_after = stuck_after
        self.base_bid = model.body("base").id
        rb = set()
        for nm in ("base", "cam_mast", "cam_tilt", "left_wheel", "right_wheel"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
            if bid >= 0:
                rb.add(bid)
        self.robot_g = set(g for g in range(model.ngeom) if model.geom_bodyid[g] in rb)
        self.obs_g = set(field.gid)
        self.reset()

    # --- utilidades ---
    def _xy(self):
        return self.d.sensor("base_pos").data[:2].copy()

    def _upright(self):
        up = self.d.xmat[self.base_bid].reshape(3, 3)[:, 2]
        return float(np.arccos(np.clip(up[2], -1, 1))) <= self.flip_thresh

    def remaining_track_distance_m(self):
        """metros que faltan SOBRE LA PISTA para terminar (no euclidiana)."""
        return float(max(self.track.length - max(self.S, 0.0), 0.0))

    def _obstacle_contacts(self):
        cur = set()
        for c in self.d.contact[:self.d.ncon]:
            g1, g2 = c.geom1, c.geom2
            if g1 in self.obs_g and g2 in self.robot_g:
                cur.add(g1)
            elif g2 in self.obs_g and g1 in self.robot_g:
                cur.add(g2)
        return cur

    def reset(self):
        _, s, lat, _ = self.track.project(self._xy())
        self.prev_s = s
        self.S = 0.0                       # progreso firmado acumulado (m)
        self.on_track = abs(lat) <= self.track.width
        self.cool = {}                     # gid -> cooldown restante (s)
        self.stuck_t = 0.0
        self.avoid = {}                    # k -> estado de esquive
        for k in self.field.active:
            self.avoid[k] = {"near": False, "collided": False, "done": False, "S0": 0.0}
        self.comp = defaultdict(float)
        self.total = 0.0

    def step(self, dt):
        """un paso de reward (dt = segundos de sim desde el ultimo step). Devuelve
        (reward, terminal, outcome). outcome in {None,'flip','lost','goal','reverse_goal'}."""
        r = 0.0; terminal = False; outcome = None
        xy = self._xy()
        dist, s, lat, tdir = self.track.project(xy)

        # --- progreso firmado (con wrap en cerradas) ---
        ds = s - self.prev_s
        if self.track.closed:
            half = self.track.length/2
            if ds > half: ds -= self.track.length
            elif ds < -half: ds += self.track.length
        self.prev_s = s
        self.S += ds

        on = abs(lat) <= self.track.width
        upright = self._upright()

        # avance FIRMADO: adelante suma, ir en contra resta (mismo peso por metro)
        pr = self.W["progress"]*ds; r += pr; self.comp["advance"] += pr
        # paso seguro DIRECCIONAL: +safe si avanza, -safe si va en contra
        if on and upright:
            if ds > 0:
                r += self.W["safe"]; self.comp["safe"] += self.W["safe"]
            elif ds < 0:
                r -= self.W["safe"]; self.comp["safe"] -= self.W["safe"]
        # fuera / retorno
        if not on:
            r -= self.W["offtrack"]; self.comp["offtrack"] -= self.W["offtrack"]
        elif not self.on_track:
            r += self.W["ret"]; self.comp["ret"] += self.W["ret"]

        self.on_track = on

        # --- colisiones: EVENTO con cooldown por objeto ---
        for g in list(self.cool):
            self.cool[g] -= dt
            if self.cool[g] <= 0:
                del self.cool[g]
        contacts = self._obstacle_contacts()
        for g in contacts:
            if g not in self.cool:
                r -= self.W["collision"]; self.comp["collision"] -= self.W["collision"]
                self.cool[g] = self.collision_cooldown
                k = self.field.gid2k.get(g)
                if k is not None:
                    self.avoid.setdefault(k, {"near": False, "collided": False, "done": False, "S0": 0.0})
                    self.avoid[k]["collided"] = True

        # --- estancamiento (acotado por segundo) ---
        if contacts and abs(ds) < 0.002:
            self.stuck_t += dt
            if self.stuck_t >= self.stuck_after:
                pen = self.W["stuck"]*dt; r -= pen; self.comp["stuck"] -= pen
        else:
            self.stuck_t = 0.0

        # --- esquive: una vez por obstaculo ---
        for k in self.field.active:
            st = self.avoid.setdefault(k, {"near": False, "collided": False, "done": False, "S0": 0.0})
            if st["done"] or st["collided"]:
                continue
            od = np.hypot(*(xy - self.field.obs_xy(k)))
            if od < self.near_radius:
                if not st["near"]:
                    st["near"] = True; st["S0"] = self.S
            elif st["near"]:
                if self.S - st["S0"] > 0.20:                 # paso de largo hacia adelante
                    r += self.W["avoid"]; self.comp["avoid"] += self.W["avoid"]; st["done"] = True
                else:
                    st["near"] = False

        # --- vuelco -> terminal ---
        if not upright:
            r -= self.W["flip"]; self.comp["flip"] -= self.W["flip"]
            terminal = True; outcome = "flip"

        # --- demasiado lejos de la pista -> terminal (sin penalidad extra) ---
        if not terminal and abs(lat) > self.far_lost:
            terminal = True; outcome = "lost"

        # --- llegada (progreso casi completo) ---
        if not terminal:
            if self.S >= self.track.length*0.95:
                terminal = True; outcome = "goal"
            elif self.S <= -self.track.length*0.95:
                terminal = True; outcome = "reverse_goal"

        self.total += r
        return r, terminal, outcome

    # --- recompensas terminales ---
    def arrival_reward(self, sim_time, time_max):
        bonus = self.W["speed_bonus"]*max(0.0, (time_max-sim_time))/time_max
        self.comp["arrival"] += self.W["arrival"]; self.comp["speed_bonus"] += bonus
        r = self.W["arrival"]+bonus; self.total += r; return r

    def reverse_reward(self):
        r = -self.W["wrong_dir"]; self.comp["wrong_dir"] -= self.W["wrong_dir"]
        self.total += r; return r

    def timeout_reward(self):
        rem = self.remaining_track_distance_m()
        r = -(self.W["timeout"] + self.W["timeout_per_m"]*rem)
        self.comp["timeout"] += r; self.total += r
        return r, rem

    def summary(self):
        c = self.comp
        return dict(total=round(self.total, 3),
                    advance=round(c["advance"], 3), safe=round(c["safe"], 3),
                    arrival=round(c["arrival"], 3), speed_bonus=round(c["speed_bonus"], 3),
                    wrong_dir=round(c["wrong_dir"], 3), timeout=round(c["timeout"], 3),
                    collision=round(c["collision"], 3), stuck=round(c["stuck"], 3),
                    avoid=round(c["avoid"], 3), offtrack=round(c["offtrack"], 3),
                    ret=round(c["ret"], 3), flip=round(c["flip"], 3))
