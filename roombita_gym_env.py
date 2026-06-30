"""
roombita_gym_env.py - entorno Gymnasium para entrenar la roombita con PPO.

ACCION (action_space): Box(-1, 1, shape=(2,))  -> [v, w]
  v = velocidad LINEAL normalizada  (-1..1 -> -MAX_V..MAX_V m/s)   -> moverse recto
  w = velocidad ANGULAR normalizada (-1..1 -> -MAX_W..MAX_W rad/s) -> girar / curva
  Es el comando (v, w) tipo Twist que recibe el robot REAL. Adentro se convierte a
  velocidades de rueda con la cinematica diferencial, asi el policy saca EXACTAMENTE
  el mismo comando que se le manda al robot fisico (interfaz de control sim-to-real).

OBSERVACION (obs_mode):
  - "state"   : vector compacto (lat, err, v, w, on_track, rem, d_obs, b_obs).
        *** OJO: es INFORMACION PRIVILEGIADA de MuJoCo (posicion exacta sobre la
        pista, distancia real a obstaculos, etc.). El robot FISICO NO tiene acceso
        a esto -> "state" NO es sim-to-real directo. Sirve solo para debug, baseline
        o un entrenamiento idealizado/rapido. Usa MlpPolicy. ***
  - "mask"    : mapa de segmentacion 3-clases (HxWx3) -> CnnPolicy.  [SIM-TO-REAL]
  - "rgb"     : camara RGB reducida (HxWx3)            -> CnnPolicy.  [SIM-TO-REAL]
  - "obstacle": mascara binaria de obstaculos (HxWx3)  -> CnnPolicy.  [SIM-TO-REAL]
  mask/rgb/obstacle son el camino realista (solo dependen de la camara), y son los
  que se usan con el segmentador propio (vision_segmenter.py) para sim-to-real.

REWARD: usa arena_env.Rewards (progreso, llegada/contraria, choque con cooldown,
esquive, salida/retorno, vuelco, timeout por distancia restante sobre la pista).

terminated = goal / reverse_goal / flip / lost ;  truncated = timeout (time_max).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from tracks import get_track
from arena_build import build_arena
from arena_env import Vision, ObstacleField, Lighting, Episode, Dynamics, Rewards

MAX_V = 0.6        # m/s   velocidad LINEAL maxima
MAX_W = 3.0        # rad/s velocidad ANGULAR maxima
WHEEL_R, HALF_AXLE = 0.028, 0.085
WHEEL_MAX = 22.0   # tope del actuador (ctrlrange del XML)


class RoombitaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, track="simple_line", obs_mode="state", obstacle_slots=8,
                 obstacle_prob=0.6, obstacle_count_mode="probabilistic", time_max=45.0,
                 brightness=1.0, random_brightness=False, cam_w=84, cam_h=84,
                 frame_skip=12, seed=None, render_mode=None):
        super().__init__()
        assert obs_mode in ("state", "mask", "rgb", "obstacle")
        self.obs_mode = obs_mode
        self.track_name = track
        self.time_max = time_max
        self.frame_skip = frame_skip
        self.obstacle_slots = obstacle_slots
        self.obstacle_prob = obstacle_prob
        self.obstacle_count_mode = obstacle_count_mode
        self.brightness = brightness
        self.random_brightness = random_brightness
        self.render_mode = render_mode
        self._seed = seed if seed is not None else 0
        self.cam_w, self.cam_h = cam_w, cam_h

        self.track = get_track(track, seed=self._seed)
        build_arena(self.track, n_slots=max(obstacle_slots, 1), path="arena.xml", seed=self._seed)
        self.m = mujoco.MjModel.from_xml_path("arena.xml")
        self.d = mujoco.MjData(self.m)
        self.dt = self.m.opt.timestep
        self.lid = self.m.actuator("left_motor").id
        self.rid = self.m.actuator("right_motor").id

        self.field = ObstacleField(self.m, self.d, self.track)
        self.light = Lighting(self.m)
        self.dyn = Dynamics(self.m, self.d, seed=self._seed)
        self.ep = Episode(self.m, self.d, self.track, seed=self._seed)
        self.rew = Rewards(self.m, self.d, self.track, self.field)
        self.base_bid = self.m.body("base").id
        self.qadr = self.ep.qadr; self.dadr = self.ep.dadr

        self.vis = None
        if obs_mode in ("mask", "rgb", "obstacle") or render_mode == "rgb_array":
            self.vis = Vision(self.m, cam_h, cam_w)

        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        if obs_mode == "state":
            self.observation_space = spaces.Box(-5.0, 5.0, (10,), np.float32)
        else:
            self.observation_space = spaces.Box(0, 255, (cam_h, cam_w, 3), np.uint8)
        self._t = 0.0

    # ---------------- estado interno ----------------
    def _robot(self):
        xy = self.d.sensor("base_pos").data[:2].copy()
        q = self.d.sensor("base_quat").data
        yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
        v = float(np.hypot(*self.d.qvel[self.dadr:self.dadr+2]))
        w = float(self.d.qvel[self.dadr+5])
        return xy, float(yaw), v, w

    def _nearest_obstacle(self, xy, yaw):
        best_d, best_b = 3.0, 0.0
        for k in self.field.active:
            o = self.field.obs_xy(k); vec = o-xy; dd = np.hypot(*vec)
            if dd < best_d:
                best_d = dd; best_b = np.arctan2(vec[1], vec[0])-yaw
        return best_d, best_b

    def _state_obs(self):
        xy, yaw, v, w = self._robot()
        _, s, lat, tdir = self.track.project(xy)
        err = np.arctan2(tdir[1], tdir[0])-yaw
        on = 1.0 if abs(lat) <= self.track.width else 0.0
        rem = self.rew.remaining_track_distance_m()/max(self.track.length, 1e-6)
        d_obs, b_obs = self._nearest_obstacle(xy, yaw)
        return np.array([np.clip(lat/0.5, -3, 3), np.sin(err), np.cos(err),
                         np.clip(v/0.6, -3, 3), np.clip(w/3, -3, 3), on, rem,
                         np.clip(d_obs/1.5, 0, 3), np.sin(b_obs), np.cos(b_obs)], np.float32)

    def _obs(self):
        if self.obs_mode == "state":
            return self._state_obs()
        if self.obs_mode == "mask":
            return self.vis.seg_map(self.d)
        if self.obs_mode == "obstacle":
            mk = self.vis.mask(self.d, "obstacle").astype(np.uint8)*255
            return np.repeat(mk[..., None], 3, axis=2)
        return self.vis.rgb(self.d)

    # ---------------- API gym ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.m, self.d)        # estado fisico limpio (determinista)
        # spawn de obstaculos y dinamica comparten el RNG seedeable de Gymnasium:
        # reproducible si pasas seed, variado entre episodios si no.
        self.ep.rng = self.np_random
        self.dyn.rng = self.np_random
        self.ep.reset(field=self.field, obstacles=(self.obstacle_slots > 0),
                      prob=self.obstacle_prob, mode=self.obstacle_count_mode,
                      lighting=self.light, brightness=self.brightness,
                      random_brightness=self.random_brightness)
        self.dyn.reset(); self.rew.reset(); self._t = 0.0
        return self._obs(), {"track": self.track_name}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        v = float(a[0])*MAX_V                          # velocidad lineal (m/s)
        w = float(a[1])*MAX_W                          # velocidad angular (rad/s)
        wl = (v - w*HALF_AXLE)/WHEEL_R                 # cinematica diferencial -> ruedas
        wr = (v + w*HALF_AXLE)/WHEEL_R
        self.d.ctrl[self.lid] = float(np.clip(wl, -WHEEL_MAX, WHEEL_MAX))
        self.d.ctrl[self.rid] = float(np.clip(wr, -WHEEL_MAX, WHEEL_MAX))
        for _ in range(self.frame_skip):
            self.dyn.update(self.dt); mujoco.mj_step(self.m, self.d); self._t += self.dt

        r, terminal, oc = self.rew.step(self.frame_skip*self.dt)
        terminated = False; truncated = False
        if oc == "goal":
            r += self.rew.arrival_reward(self._t, self.time_max); terminated = True
        elif oc == "reverse_goal":
            r += self.rew.reverse_reward(); terminated = True
        elif oc in ("flip", "lost"):
            terminated = True
        elif self._t >= self.time_max:
            tr, _ = self.rew.timeout_reward(); r += tr; truncated = True

        info = {"outcome": oc or ("timeout" if truncated else "running"),
                "sim_time": round(self._t, 2)}
        if terminated or truncated:
            info["episode_reward"] = round(self.rew.total, 3)
            info["components"] = self.rew.summary()
        return self._obs(), float(r), terminated, truncated, info

    def render(self):
        if self.vis is None:
            self.vis = Vision(self.m, self.cam_h, self.cam_w)
        return self.vis.rgb(self.d)

    def close(self):
        pass
