"""
seg_model.py - red de segmentacion propia (MiniUNet), liviana y entrenable en CPU/GPU.
Entrada [B,3,H,W] -> salida [B,num_classes,H,W] (segmentacion pixel-wise).
H y W deben ser divisibles por 4 (dos poolings). img_size=128 funciona.

PALETTE: colores por clase, iguales a los del seg_map del simulador, para que la
mascara predicha tenga la MISMA distribucion que la que vio PPO en obs_mode=mask.
"""
import torch
import torch.nn as nn

# 0=fondo  (resto segun num_classes)
PALETTE = {
    2: [(0, 0, 0), (235, 40, 40)],                                   # obstacle
    3: [(0, 0, 0), (90, 90, 100), (235, 40, 40)],                    # combined: bg/pista/obstaculo
    4: [(0, 0, 0), (90, 90, 100), (240, 215, 15), (235, 40, 40)],    # seg: bg/road/line/obstacle
}


def _cbr(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class MiniUNet(nn.Module):
    def __init__(self, num_classes=3, ch=(16, 32, 64)):
        super().__init__()
        c1, c2, c3 = ch
        self.num_classes = num_classes
        self.e1 = _cbr(3, c1); self.e2 = _cbr(c1, c2); self.e3 = _cbr(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2); self.d2 = _cbr(c2*2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2); self.d1 = _cbr(c1*2, c1)
        self.head = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


def colorize(labels, num_classes):
    """labels (H,W) int -> imagen RGB (H,W,3) uint8 con la paleta del simulador."""
    import numpy as np
    pal = np.array(PALETTE.get(num_classes, PALETTE[3]), np.uint8)
    return pal[np.clip(labels, 0, num_classes-1)]
