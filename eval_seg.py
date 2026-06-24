"""
eval_seg.py - evalua un segmentador entrenado sobre clips .npz.

  python eval_seg.py --model models/seg_unet.pth --data_dir runs/

Reporta loss, pixel accuracy e IoU por clase (fondo / pista / obstaculo ...).
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seg_dataset import SegClipDataset
from seg_model import MiniUNet
from train_seg import evaluate

NAMES = {2: ["fondo", "obstaculo"],
         3: ["fondo", "pista", "obstaculo"],
         4: ["fondo", "road", "line", "obstaculo"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", default="runs/")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_frames_per_clip", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location=args.device, weights_only=False)
    n = ck["num_classes"]; target = ck["target"]; img = ck.get("img_size", 128)
    model = MiniUNet(num_classes=n).to(args.device)
    model.load_state_dict(ck["state_dict"])

    ds = SegClipDataset(args.data_dir, target=target, img_size=img,
                        max_frames_per_clip=args.max_frames_per_clip)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    loss, acc, iou = evaluate(model, dl, n, args.device, nn.CrossEntropyLoss())
    names = NAMES.get(n, [f"c{i}" for i in range(n)])
    print(f"[eval] {len(ds)} frames | target={target} | loss={loss:.3f} | pixel_acc={acc:.3f}")
    for i in range(n):
        print(f"   IoU {names[i]:10s} = {iou[i]:.3f}")
    print(f"   mIoU = {iou.mean():.3f}")


if __name__ == "__main__":
    main()
