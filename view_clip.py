"""
view_clip.py - reproductor de clips grabados por run_play.py.

  python view_clip.py --clip runs/2026-06-17_22-35-12/episode_0001.npz

Muestra lado a lado RGB | segmentacion | obstaculos, con el reward por frame y
acumulado. Controles:
  ESPACIO  play / pausa
  . o ->   avanzar un frame (en pausa)
  , o <-   retroceder un frame (en pausa)
  r        reiniciar
  q o ESC  salir
"""
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="ruta al .npz del episodio")
    ap.add_argument("--scale", type=int, default=3, help="zoom de los frames")
    ap.add_argument("--fps", type=float, default=0, help="0 = usar fps del clip")
    args = ap.parse_args()

    import cv2
    z = np.load(args.clip, allow_pickle=True)
    rgb, seg, obs = z["rgb"], z["seg"], z["obstacle"]
    reward = z["reward"] if "reward" in z.files else np.zeros(len(rgb), np.float32)
    done = z["done"] if "done" in z.files else np.zeros(len(rgb), bool)
    info = json.loads(str(z["info"])) if "info" in z.files else {}
    track = str(z["track_name"]) if "track_name" in z.files else "?"
    fps = args.fps or float(info.get("fps", 15))
    cum = np.cumsum(reward)
    T, h, w = len(rgb), rgb.shape[1], rgb.shape[2]
    sc = args.scale
    print(f"clip: {track} | {T} frames | {fps:.1f} fps | outcome={info.get('outcome','?')} "
          f"| reward total={cum[-1]:.2f}")
    print("ESPACIO play/pausa | . avanzar | , retroceder | r reiniciar | q salir")

    panelW = w*sc
    def render(i):
        trio = np.hstack([rgb[i], seg[i], obs[i]])
        big = cv2.resize(cv2.cvtColor(trio, cv2.COLOR_RGB2BGR), (panelW*3, h*sc),
                         interpolation=cv2.INTER_NEAREST)
        bar = np.full((70, panelW*3, 3), 25, np.uint8)
        # curva de reward acumulado
        if T > 1:
            xs = (np.arange(T)/(T-1)*(panelW*3-1)).astype(int)
            ys = (60 - (cum-cum.min())/(np.ptp(cum)+1e-6)*52).astype(int)+4
            for j in range(1, T):
                cv2.line(bar, (xs[j-1], ys[j-1]), (xs[j], ys[j]), (90, 200, 90), 1)
            cv2.line(bar, (xs[i], 2), (xs[i], 68), (60, 60, 230), 1)
        canvas = np.vstack([big, bar])
        col = (60, 120, 240) if done[i] else (0, 235, 0)
        cv2.putText(canvas, f"{track}  f{i+1}/{T}  r={reward[i]:+.2f}  acum={cum[i]:+.2f}"
                            f"   [RGB | seg | obstaculos]", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        return canvas

    cv2.namedWindow("clip", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("clip", panelW*3, h*sc+70)
    i, playing = 0, True
    delay = max(int(1000/fps), 1)
    while True:
        cv2.imshow("clip", render(i))
        k = cv2.waitKey(delay if playing else 0) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            playing = not playing
        elif k in (ord("."), 83):
            playing = False; i = min(i+1, T-1)
        elif k in (ord(","), 81):
            playing = False; i = max(i-1, 0)
        elif k == ord("r"):
            i = 0
        elif playing:
            i += 1
            if i >= T:
                i = T-1; playing = False
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
