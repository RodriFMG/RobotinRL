"""
train_seg.py - entrena el segmentador propio (MiniUNet) con los clips .npz.

  python train_seg.py --data_dir runs/ --target combined --epochs 20 \
         --batch_size 16 --lr 1e-3 --img_size 128 --save_path models/seg_unet.pth

Guarda: modelo .pth (con config y num_classes), y metricas (loss, pixel acc, IoU/clase).
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from seg_dataset import SegClipDataset
from seg_model import MiniUNet


def confusion(pred, gt, n):
    k = (gt >= 0) & (gt < n)
    return np.bincount(n*gt[k].astype(int) + pred[k], minlength=n*n).reshape(n, n)


def metrics_from_conf(c):
    inter = np.diag(c).astype(float)
    union = c.sum(1) + c.sum(0) - inter
    iou = inter / np.maximum(union, 1e-9)
    acc = inter.sum() / max(c.sum(), 1e-9)
    return acc, iou


@torch.no_grad()
def evaluate(model, loader, n, device, crit):
    model.eval(); conf = np.zeros((n, n), int); tot = 0.0; nb = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x); tot += crit(out, y).item(); nb += 1
        conf += confusion(out.argmax(1).cpu().numpy().ravel(), y.cpu().numpy().ravel(), n)
    acc, iou = metrics_from_conf(conf)
    return tot/max(nb, 1), acc, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="runs/")
    ap.add_argument("--target", default="combined", choices=["seg", "obstacle", "combined"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--max_frames_per_clip", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--save_path", default="models/seg_unet.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ds = SegClipDataset(args.data_dir, target=args.target, img_size=args.img_size,
                        max_frames_per_clip=args.max_frames_per_clip)
    n = ds.num_classes
    nv = max(1, int(len(ds)*args.val_frac)); nt = len(ds)-nv
    tr, va = random_split(ds, [nt, nv], generator=torch.Generator().manual_seed(0))
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[data] {len(ds)} frames | train {nt} / val {nv} | clases={n} ({args.target})")

    dev = args.device
    model = MiniUNet(num_classes=n).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best = 0.0; hist = []
    for ep in range(1, args.epochs+1):
        model.train(); run = 0.0; nb = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); out = model(x); loss = crit(out, y)
            loss.backward(); opt.step(); run += loss.item(); nb += 1
        vloss, vacc, viou = evaluate(model, vl, n, dev, crit)
        miou = float(viou.mean())
        iou_str = " ".join(f"c{c}={viou[c]:.2f}" for c in range(n))
        print(f"ep {ep:2d}/{args.epochs} | train_loss {run/max(nb,1):.3f} | "
              f"val_loss {vloss:.3f} | acc {vacc:.3f} | mIoU {miou:.3f} | {iou_str}")
        hist.append(dict(epoch=ep, train_loss=run/max(nb, 1), val_loss=vloss,
                         pixel_acc=vacc, miou=miou, iou=[float(v) for v in viou]))
        if miou >= best:
            best = miou
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "num_classes": n,
                        "target": args.target, "img_size": args.img_size,
                        "config": vars(args)}, args.save_path)
    with open(os.path.splitext(args.save_path)[0]+"_metrics.json", "w") as f:
        json.dump({"config": vars(args), "history": hist, "best_miou": best}, f, indent=2)
    print(f"[ok] mejor mIoU={best:.3f} | modelo en {args.save_path}")


if __name__ == "__main__":
    main()
