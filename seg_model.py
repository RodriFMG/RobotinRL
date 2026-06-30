"""
seg_model.py - red de segmentacion propia (MiniUNet), liviana y entrenable en CPU/GPU.
Entrada [B,3,H,W] -> salida [B,num_classes,H,W] (segmentacion pixel-wise).
H y W deben ser divisibles por 4 (dos poolings). img_size=128 funciona.

PALETTE: colores por clase, iguales a los del seg_map del simulador, para que la
mascara predicha tenga la MISMA distribucion que la que vio PPO en obs_mode=mask.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PALETTE = {
    2: [(0,0,0), (235,40,40)],                                # obstacle
    3: [(0,0,0), (90,90,100), (235,40,40)],                   # combined: bg/pista/obstaculo
    4: [(0,0,0), (90,90,100), (240,215,15), (235,40,40)],     # seg: bg/road/line/obstacle
}

def _cbr(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class MiniUNet(nn.Module):
    def __init__(self, num_classes=3, ch=(16, 32, 64)):
        super().__init__()
        self.num_classes = num_classes
        c1, c2, c3 = ch[0], ch[1], ch[2]
        c4, c5 = c3*2, c3*4                       # niveles extra derivados de ch
        self.e1 = _cbr(3, c1)
        self.e2 = _cbr(c1, c2)
        self.e3 = _cbr(c2, c3)
        self.e4 = _cbr(c3, c4)
        self.bott = _cbr(c4, c5)                  # bottleneck (embedding mas chico)
        self.drop = nn.Dropout2d(0.1)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(c5, c4, 2, 2); self.d4 = _cbr(c4*2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, 2); self.d3 = _cbr(c3*2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2); self.d2 = _cbr(c2*2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2); self.d1 = _cbr(c1*2, c1)
        self.head = nn.Conv2d(c1, num_classes, 1)

    def _up(self, up, dec, x, skip):
        x = up(x)
        if x.shape[-2:] != skip.shape[-2:]:       # robusto a tamanos no divisibles por 16
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return dec(torch.cat([x, skip], 1))

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.drop(self.bott(self.pool(e4)))   # embedding a H/16
        d4 = self._up(self.up4, self.d4, b,  e4)
        d3 = self._up(self.up3, self.d3, d4, e3)
        d2 = self._up(self.up2, self.d2, d3, e2)
        d1 = self._up(self.up1, self.d1, d2, e1)
        return self.head(d1)

def colorize(labels, num_classes):
    pal = np.array(PALETTE.get(num_classes, PALETTE[3]), np.uint8)
    return pal[np.clip(labels, 0, num_classes-1)]