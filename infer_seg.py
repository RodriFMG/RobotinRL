"""
infer_seg.py - corre el segmentador sobre un clip y compara RGB | real | predicha.

  python infer_seg.py --model models/seg_unet.pth --clip runs/.../episode_0001.npz
  python infer_seg.py --model models/seg_unet.pth --clip ... --frame 30 --save out.png

Guarda una imagen comparativa (y la muestra si hay display).
"""
import argparse
import numpy as np
import torch

from seg_model import MiniUNet, colorize
from seg_dataset import seg_rgb_to_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frame", type=int, default=-1, help="-1 = varios frames")
    ap.add_argument("--save", default="seg_infer.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location=args.device, weights_only=False)
    n = ck["num_classes"]; target = ck["target"]; img = ck.get("img_size", 128)
    model = MiniUNet(num_classes=n).to(args.device).eval()
    model.load_state_dict(ck["state_dict"])

    z = np.load(args.clip, allow_pickle=True)
    rgb, seg = z["rgb"], z["seg"]
    T = len(rgb)
    frames = [args.frame] if args.frame >= 0 else list(np.linspace(0, T-1, min(T, 4)).astype(int))

    import torch.nn.functional as F
    rows = []
    for i in frames:
        x = torch.from_numpy(np.ascontiguousarray(rgb[i])).permute(2, 0, 1).float()/255.0
        x = F.interpolate(x[None], size=(img, img), mode="bilinear", align_corners=False).to(args.device)
        with torch.no_grad():
            pred = model(x).argmax(1)[0].cpu().numpy()
        gt = seg_rgb_to_label(seg[i], target)
        gt_small = F.interpolate(torch.from_numpy(gt)[None, None].float(),
                                 size=(img, img), mode="nearest")[0, 0].long().numpy()
        rgb_small = F.interpolate(torch.from_numpy(np.ascontiguousarray(rgb[i])).permute(2, 0, 1)[None].float(),
                                  size=(img, img))[0].permute(1, 2, 0).byte().numpy()
        rows.append(np.hstack([rgb_small, colorize(gt_small, n), colorize(pred, n)]))
    grid = np.vstack(rows)

    try:
        from PIL import Image
        Image.fromarray(grid).save(args.save)
        print(f"[ok] comparativa guardada en {args.save}  (cols: RGB | real | predicha)")
    except Exception as e:
        print("no pude guardar PNG:", e)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 3*len(frames))); plt.imshow(grid); plt.axis("off")
        plt.title("RGB | mascara real | mascara predicha"); plt.tight_layout(); plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
