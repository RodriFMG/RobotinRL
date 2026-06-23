"""
tracks.py - catalogo de pistas + clase Track.

Cada pista es una polilinea de waypoints (x,y) en el plano del piso. La clase
Track proyecta la posicion del robot sobre la polilinea y entrega:
  - s     : longitud de arco recorrida (m) desde el inicio
  - lat   : distancia lateral con signo al centro de la pista (m)
  - tdir  : tangente (direccion correcta de avance) en ese punto
Esto permite medir PROGRESO y DIRECCION sobre CUALQUIER pista del catalogo.

  get_track(name, seed=0) -> Track
  TRACKS = lista de nombres disponibles
"""
import numpy as np

TRACKS = ["straight", "simple_line", "oval", "zigzag", "s_curve",
          "eight", "hairpin", "complex", "random"]


class Track:
    def __init__(self, pts, closed=True, name="track", width=0.20):
        self.name = name
        self.closed = bool(closed)
        self.width = float(width)
        pts = np.asarray(pts, float)
        # quitar puntos repetidos consecutivos
        keep = [0] + [i for i in range(1, len(pts)) if np.hypot(*(pts[i]-pts[i-1])) > 1e-4]
        self.pts = pts[keep]
        # en cerradas, si el ultimo coincide con el primero, sacarlo (evita segmento nulo)
        if self.closed and len(self.pts) > 2 and np.hypot(*(self.pts[0]-self.pts[-1])) < 1e-4:
            self.pts = self.pts[:-1]
        seg = np.diff(self.pts, axis=0)
        if self.closed:
            seg = np.vstack([seg, self.pts[0]-self.pts[-1]])
        self.seg = seg
        self.seglen = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seglen)])
        self.length = float(self.cum[-1])
        self.start = self.pts[0].copy()
        self.start_yaw = float(np.arctan2(seg[0, 1], seg[0, 0]))
        self.finish = (self.pts[0].copy() if self.closed else self.pts[-1].copy())

    def project(self, xy):
        """devuelve (dist, s, lat, tdir) del punto mas cercano de la polilinea."""
        P = np.asarray(xy, float)
        A = self.pts[:len(self.seg)]
        AB = self.seg
        L2 = (AB**2).sum(1)
        t = np.clip(((P-A)*AB).sum(1) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
        proj = A + t[:, None]*AB
        dvec = P - proj
        dist = np.hypot(dvec[:, 0], dvec[:, 1])
        i = int(dist.argmin())
        s = float(self.cum[i] + t[i]*self.seglen[i])
        tn = self.seglen[i] if self.seglen[i] > 0 else 1.0
        tdir = AB[i] / tn
        lat = float(np.cross(tdir, P - proj[i]))     # >0 a la izquierda
        return float(dist[i]), s, lat, tdir

    def road_geoms(self, road_rgba, line_rgba):
        """genera los geoms (box, a ras del piso, sin colision) de asfalto y linea."""
        road, line = [], []
        n = len(self.seg)
        for i in range(n):
            if self.seglen[i] < 1e-4:
                continue
            a = self.pts[i]; b = a + self.seg[i]
            cx, cy = (a+b)/2
            L = self.seglen[i]*1.6
            yaw = np.arctan2(self.seg[i, 1], self.seg[i, 0])
            road.append(f'    <geom type="box" pos="{cx:.3f} {cy:.3f} 0.001" euler="0 0 {yaw:.3f}" '
                        f'size="{L/2:.3f} {self.width:.3f} 0.001" rgba="{road_rgba}" '
                        f'contype="0" conaffinity="0"/>')
            line.append(f'    <geom type="box" pos="{cx:.3f} {cy:.3f} 0.0025" euler="0 0 {yaw:.3f}" '
                        f'size="{L/2:.3f} 0.022 0.0012" rgba="{line_rgba}" contype="0" conaffinity="0"/>')
        return "\n".join(road), "\n".join(line)


# ----------------------------- catalogo -----------------------------
def _closed(fn, M=200):
    t = np.linspace(0, 2*np.pi, M, endpoint=False)
    return fn(t)


def get_track(name, seed=0):
    name = (name or "simple_line").lower()
    if name == "random":
        return _random_track(seed)
    if name == "straight":
        u = np.linspace(0, 1, 90)
        pts = np.stack([-1.7+3.4*u, np.zeros_like(u)], 1)
        return Track(pts, closed=False, name="straight")
    if name == "s_curve":
        u = np.linspace(0, 1, 120)
        pts = np.stack([-1.7+3.4*u, 0.85*np.sin(2*np.pi*u)], 1)
        return Track(pts, closed=False, name="s_curve")
    if name == "simple_line":
        f = lambda t: np.stack([(1.25+0.20*np.cos(2*t))*np.cos(t),
                                (1.25+0.20*np.cos(2*t))*np.sin(t)], 1)
        return Track(_closed(f), name="simple_line")
    if name == "oval":
        f = lambda t: np.stack([1.7*np.cos(t), 1.05*np.sin(t)], 1)
        return Track(_closed(f), name="oval")
    if name == "zigzag":
        f = lambda t: np.stack([(1.2+0.30*np.cos(4*t))*np.cos(t),
                                (1.2+0.30*np.cos(4*t))*np.sin(t)], 1)
        return Track(_closed(f), name="zigzag")
    if name == "eight":
        f = lambda t: np.stack([1.6*np.cos(t), 1.6*np.sin(t)*np.cos(t)], 1)
        return Track(_closed(f), name="eight")
    if name == "hairpin":
        return _hairpin()
    if name == "complex":
        f = lambda t: np.stack([(1.2+0.18*np.cos(2*t)+0.15*np.sin(3*t)+0.12*np.cos(5*t))*np.cos(t),
                                (1.2+0.18*np.cos(2*t)+0.15*np.sin(3*t)+0.12*np.cos(5*t))*np.sin(t)], 1)
        return Track(_closed(f), name="complex")
    # fallback
    return get_track("simple_line")


def _hairpin():
    # dos rectas paralelas unidas por dos semicirculos MUY cerrados (hairpins)
    pts = []
    R = 0.25
    xs = np.linspace(-1.5, 1.3, 60)
    pts += [(x, R) for x in xs]                                   # recta superior ->
    ang = np.linspace(np.pi/2, -np.pi/2, 24)                      # giro cerrado derecho
    pts += [(1.3+R*np.cos(a), R*np.sin(a)) for a in ang]
    pts += [(x, -R) for x in xs[::-1]]                            # recta inferior <-
    ang = np.linspace(-np.pi/2, -3*np.pi/2, 24)                   # giro cerrado izquierdo
    pts += [(-1.5+R*np.cos(a), R*np.sin(a)) for a in ang]
    return Track(np.array(pts), closed=True, name="hairpin")


def _random_track(seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2*np.pi, 200, endpoint=False)
    r = np.full_like(t, 1.2)
    for k in (2, 3, 4, 5):
        r += rng.uniform(-0.22, 0.22)*np.cos(k*t + rng.uniform(0, 2*np.pi))
    r = np.clip(r, 0.7, 1.7)
    pts = np.stack([r*np.cos(t), r*np.sin(t)], 1)
    return Track(pts, name=f"random(seed={seed})")
