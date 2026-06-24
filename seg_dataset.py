"""
seg_dataset.py - dataset PyTorch a partir de los clips .npz grabados.

Busca recursivamente todos los .npz bajo data_dir, usa 'rgb' como entrada y deriva
el target desde 'seg' (el mapa de segmentacion limpio del simulador).

  --target seg       4 clases: 0 fondo, 1 pista(road), 2 linea(line), 3 obstaculo
  --target obstacle  2 clases: 0 fondo, 1 obstaculo
  --target combined  3 clases: 0 fondo, 1 pista/linea, 2 obstaculo   (recomendado)

Devuelve (x, y):
  x: RGB normalizada [3,H,W] float (0..1)
  y: mascara [H,W] long con labels enteros (para CrossEntropyLoss)
"""
import os, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

TARGETS = {"seg": 4, "obstacle": 2, "combined": 3}


def seg_rgb_to_label(seg, target):
    """seg (H,W,3) uint8 (colores del seg_map) -> labels (H,W) int64."""
    R, G, B = seg[..., 0].astype(int), seg[..., 1].astype(int), seg[..., 2].astype(int)
    obstacle = (R > 170) & (G < 110) & (B < 110)
    line = (R > 170) & (G > 150) & (B < 90)
    road = (R > 50) & (R < 140) & (G > 50) & (G < 140) & (B > 70) & (B < 150)
    lab = np.zeros(seg.shape[:2], np.int64)
    if target == "combined":
        lab[road | line] = 1; lab[obstacle] = 2
    elif target == "seg":
        lab[road] = 1; lab[line] = 2; lab[obstacle] = 3
    else:                                   # obstacle
        lab[obstacle] = 1
    return lab


class SegClipDataset(Dataset):
    def __init__(self, data_dir, target="combined", img_size=128, max_frames_per_clip=0):
        assert target in TARGETS, f"target invalido: {target}"
        self.target = target
        self.num_classes = TARGETS[target]
        self.img = img_size
        self.files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No encontre clips .npz bajo: {data_dir}")
        self.index = []
        for f in self.files:
            with np.load(f, allow_pickle=True) as z:
                T = len(z["rgb"]) if "rgb" in z.files else 0
                has_seg = "seg" in z.files
            if not has_seg or T == 0:
                continue
            sel = (np.linspace(0, T-1, max_frames_per_clip).astype(int)
                   if max_frames_per_clip and T > max_frames_per_clip else range(T))
            self.index += [(f, int(i)) for i in sel]
        if not self.index:
            raise RuntimeError("Los .npz no tienen claves 'rgb'/'seg' utilizables.")
        self._cache = {}

    def __len__(self):
        return len(self.index)

    def _load(self, f):
        if f not in self._cache:
            if len(self._cache) > 3:
                self._cache.clear()
            z = np.load(f, allow_pickle=True)
            self._cache[f] = (z["rgb"], z["seg"])
        return self._cache[f]

    def __getitem__(self, k):
        f, i = self.index[k]
        rgb, seg = self._load(f)
        x = torch.from_numpy(np.ascontiguousarray(rgb[i])).permute(2, 0, 1).float()/255.0
        y = torch.from_numpy(seg_rgb_to_label(seg[i], self.target))
        x = F.interpolate(x[None], size=(self.img, self.img), mode="bilinear", align_corners=False)[0]
        y = F.interpolate(y[None, None].float(), size=(self.img, self.img), mode="nearest")[0, 0].long()
        return x, y
