"""
vision_segmenter.py - usa el segmentador propio para convertir RGB -> mascara.

Flujo final sim-to-real:
   RGB (sim o real)  ->  VisionSegmenter  ->  mascara predicha (seg_map)  ->  PPO(mask)  ->  ruedas

  seg = VisionSegmenter("models/seg_unet.pth")
  label_mask = seg.predict_mask(rgb_frame)      # (H,W) labels
  seg_map    = seg.predict_seg_map(rgb_frame)   # (H,W,3) uint8, paleta del simulador
"""
import numpy as np
import torch
import torch.nn.functional as F

from seg_model import MiniUNet, colorize


class VisionSegmenter:
    def __init__(self, model_path, device=None, out_size=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(model_path, map_location=self.device, weights_only=False)
        self.num_classes = ck["num_classes"]
        self.target = ck["target"]
        self.img = ck.get("img_size", 128)
        self.out_size = out_size                       # (H,W) de salida; None = como entra
        self.model = MiniUNet(num_classes=self.num_classes).to(self.device).eval()
        self.model.load_state_dict(ck["state_dict"])

    @torch.no_grad()
    def predict_mask(self, rgb_frame):
        """rgb_frame (H,W,3) uint8 -> labels (Ho,Wo) int."""
        H, W = rgb_frame.shape[:2]
        oh, ow = self.out_size or (H, W)
        x = torch.from_numpy(np.ascontiguousarray(rgb_frame)).permute(2, 0, 1).float()/255.0
        x = F.interpolate(x[None], size=(self.img, self.img), mode="bilinear",
                          align_corners=False).to(self.device)
        logits = self.model(x)
        lab = F.interpolate(logits, size=(oh, ow), mode="bilinear",
                            align_corners=False).argmax(1)[0].cpu().numpy()
        return lab

    def predict_seg_map(self, rgb_frame):
        """rgb_frame -> mascara RGB (Ho,Wo,3) uint8 con la paleta del simulador
        (misma distribucion que obs_mode=mask para alimentar a PPO)."""
        return colorize(self.predict_mask(rgb_frame), self.num_classes)
