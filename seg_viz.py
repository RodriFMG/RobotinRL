"""
seg_viz.py - visualizacion para testear el segmentador.

DOS MODOS:

1) curva de loss/metricas (lee el *_metrics.json):
   python seg_viz.py loss --metrics seg_dice_metrics.json

2) reproductor de clip + GRABACION de video:
   muestra y guarda  CLIP ORIGINAL | MASKED | SEGMENT MASKED
   python seg_viz.py clip --model seg_dice.pth --clip runs/.../episode_0001.npz
   -> guarda seg_video/video_<fecha>.mp4   (para compartir con el team)
   Controles: ESPACIO play/pausa | . avanzar | , retroceder | r reinicio | q salir
   (con --no_show solo graba el video, sin ventana)
"""
import argparse, json, os, datetime
import numpy as np


# ----------------------------- modo 1: loss -----------------------------
def plot_loss(args):
    import matplotlib.pyplot as plt
    with open(args.metrics) as f:
        m = json.load(f)
    h = m["history"]
    ep = [r["epoch"] for r in h]
    tr = [r["train_loss"] for r in h]
    vl = [r["val_loss"] for r in h]
    miou = [r["miou"] for r in h]
    acc = [r["pixel_acc"] for r in h]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, tr, label="train loss"); a1.plot(ep, vl, label="val loss")
    a1.set_xlabel("epoca"); a1.set_ylabel("loss"); a1.set_title("Loss"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(ep, miou, label="mIoU"); a2.plot(ep, acc, label="pixel acc")
    if h and "iou" in h[0]:
        iou = np.array([r["iou"] for r in h])
        for c in range(iou.shape[1]):
            a2.plot(ep, iou[:, c], "--", alpha=.6, label=f"IoU c{c}")
    a2.set_xlabel("epoca"); a2.set_ylabel("metric"); a2.set_title("mIoU / accuracy")
    a2.legend(fontsize=8); a2.grid(alpha=.3)
    fig.suptitle(f"best mIoU = {m.get('best_miou', 0):.3f}")
    fig.tight_layout(); fig.savefig(args.save, dpi=110)
    print(f"[ok] curva guardada en {args.save}")
    try:
        plt.show()
    except Exception:
        pass


# ----------------------------- modo 2: clip + video -----------------------------
def view_clip(args):
    import cv2
    from vision_segmenter import VisionSegmenter

    z = np.load(args.clip, allow_pickle=True)
    rgb = z["rgb"]
    seg_real = z["seg"] if "seg" in z.files else None
    info = json.loads(str(z["info"])) if "info" in z.files else {}
    track = str(z["track_name"]) if "track_name" in z.files else "?"
    T, h, w = len(rgb), rgb.shape[1], rgb.shape[2]
    seg = VisionSegmenter(args.model, out_size=(h, w))
    fps = args.fps or float(info.get("fps", 15))

    vsc = max(1, round(args.panel_w / w))           # cada panel ~panel_w px de ancho
    pw, ph = w*vsc, h*vsc
    HEAD = max(30, ph//12)
    titles = ["CLIP ORIGINAL", "MASKED", "SEGMENT MASKED"]
    print(f"clip: {track} | {T} frames | {fps:.1f} fps | modelo {args.model}")

    def panel(i):
        real = seg_real[i] if seg_real is not None else np.zeros_like(rgb[i])
        pred = seg.predict_seg_map(rgb[i])                       # RGB -> mascara del MODELO
        trio = np.hstack([rgb[i], real, pred])
        big = cv2.resize(cv2.cvtColor(trio, cv2.COLOR_RGB2BGR), (pw*3, ph),
                         interpolation=cv2.INTER_NEAREST)
        head = np.full((HEAD, pw*3, 3), 30, np.uint8)
        for c, ttl in enumerate(titles):
            (tw, _), _ = cv2.getTextSize(ttl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(head, ttl, (c*pw + (pw-tw)//2, HEAD-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 235, 0), 2)
        # separadores entre columnas
        for c in (1, 2):
            cv2.line(head, (c*pw, 0), (c*pw, HEAD), (80, 80, 80), 1)
            cv2.line(big, (c*pw, 0), (c*pw, ph), (80, 80, 80), 1)
        canvas = np.vstack([head, big])
        cv2.putText(canvas, f"{track}  f{i+1}/{T}", (8, HEAD+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
        return canvas

    # --- grabar el video ---
    os.makedirs("seg_video", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join("seg_video", f"video_{stamp}.mp4")
    W, H = pw*3, ph + HEAD
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    frames = []
    for i in range(T):
        f = panel(i); frames.append(f); vw.write(f)
    vw.release()
    print(f"[ok] video guardado en {out_path}  ({T} frames, {W}x{H}, {fps:.1f} fps)")

    # --- viewer interactivo (opcional) ---
    if args.no_show:
        return
    cv2.namedWindow("seg clip", cv2.WINDOW_NORMAL); cv2.resizeWindow("seg clip", W, H)
    delay = max(int(1000/fps), 1)
    i, playing = 0, True
    while True:
        cv2.imshow("seg clip", frames[i])
        k = cv2.waitKey(delay if playing else 0) & 0xFF
        if k in (ord("q"), 27): break
        elif k == ord(" "): playing = not playing
        elif k in (ord("."), 83): playing = False; i = min(i+1, T-1)
        elif k in (ord(","), 81): playing = False; i = max(i-1, 0)
        elif k == ord("r"): i = 0
        elif playing:
            i += 1
            if i >= T: i = T-1; playing = False
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("loss")
    p1.add_argument("--metrics", default="seg_dice_metrics.json")
    p1.add_argument("--save", default="seg_loss.png")

    p2 = sub.add_parser("clip")
    p2.add_argument("--model", required=True)
    p2.add_argument("--clip", required=True)
    p2.add_argument("--panel_w", type=int, default=560, help="ancho aprox de cada panel en el video")
    p2.add_argument("--fps", type=float, default=0, help="0 = usar fps del clip")
    p2.add_argument("--no_show", action="store_true", help="solo graba el video, sin ventana")

    args = ap.parse_args()
    if args.mode == "loss":
        plot_loss(args)
    else:
        view_clip(args)


if __name__ == "__main__":
    main()