# recalibrate_val.py — Step 15: re-measure flared AP with module-recalibrated confidences
# Pass 1 (sanity): plain YOLOv10 on flared images — must reproduce val-4 (car 0.597, traffic light 0.0261, all 0.249)
# Pass 2 (recalibrated): car/light confidences += module-predicted drop, then re-measure AP
# Uses ultralytics' own ap_per_class so the numbers match `yolo val`.
# v3 (25 Nov 2026 ~12:36 PKT): batch 8 STILL OOM'd in Pass 1 at the same 2.75 GiB conv
# allocation, so BATCH 8 -> 4 and chunk 1000 -> 100, plus empty_cache after each chunk.
# v4 (25 Nov 2026 ~13:26 PKT): v3 fixed the OOM (all 10000 images done in Pass 1) but the
# run then crashed in ap_per_class with IndexError: tp.shape[1] — ultralytics expects tp
# 2-D (N, num_iou_thresholds), our tp was 1-D. Fix: reshape to (N, 1) = mAP50 only.
# Also saves tp/conf/pred_cls/target_cls to .npz before ap_per_class so a later crash
# never loses the inference work again.
# v5 (25 Nov 2026 ~13:35 PKT): v4's npz save WORKED (step15_plain.npz saved) but the run
# crashed again in ap_per_class with ValueError: too many values to unpack (expected 7)
# — this ultralytics version returns MORE than 7 values. Fix: never unpack by position,
# take res[5] = ap and res[6] = unique classes by index. Also resumable: if a step15
# .npz already exists, that pass is loaded from disk instead of re-running inference.
# v6 (25 Nov 2026 ~13:54 PKT): v5 RESULT — recalibration HURT: car 0.5895 -> 0.4556
# (-0.1339), traffic light 0.0242 -> 0.0107 (-0.0135). Sign was verified CORRECT
# (target = conf_clean - conf_flare, added back). Cause: train/inference mismatch —
# the module was trained on REAL detections at conf >= 0.25 (Step 13), but v5 applied
# it to ALL boxes at conf = 0.0001, including thousands of junk boxes per image.
# Boosting junk scrambled the ranking. v6 fix: only recalibrate boxes with
# conf >= 0.25. Also overfitting was already visible in Step 14 (val MSE flat
# ~0.009 while train MSE fell to 0.0008), so even v6 may not fully recover.
# v6 saves step15_recal_v6.npz (the v5 recal npz is kept, not overwritten).
import gc
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
         5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLARED_IMG = Path(r"C:\flare-yolo\flared\images\val")
FLARED_LBL = Path(r"C:\flare-yolo\flared\labels\val")
BATCH = 4  # v3: batch 8 still OOM'd in Pass 1 (2.75 GiB conv alloc) — 4 is safer on 8 GB
CHUNK = 100  # v3: was 1000 images per predict call — 100 keeps peak memory low
RECAL_MIN_CONF = 0.25  # v6: only recalibrate boxes at/above the conf Step 13 trained on


class FlareModule(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.mlp = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.mlp(self.backbone(x)).squeeze(1)

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_module(path):
    m = FlareModule()
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return m.to(DEVICE).eval()

def read_gt(label_path, W, H):
    gt = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            c = int(p[0]); cx, cy, w, h = map(float, p[1:5])
            gt.append([c, (cx - w/2) * W, (cy - h/2) * H, (cx + w/2) * W, (cy + h/2) * H])
    return np.array(gt, dtype=np.float32).reshape(-1, 5)

def box_iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)

def ap_dict(tp, conf, pred_cls, target_cls):
    # v5: ap_per_class returns 7 values in old ultralytics, 8-9 in newer ones.
    # Never unpack by position — ap is index 5, unique classes index 6 in both.
    res = ap_per_class(tp, conf, pred_cls, target_cls, names=NAMES)
    ap, unique = res[5], res[6]
    return {int(c): float(ap[j, 0]) for j, c in enumerate(unique)}

def ap_from_npz(path):
    d = np.load(path)
    return ap_dict(d["tp"], d["conf"], d["pred_cls"], d["target_cls"])

def evaluate(model, modules, recalibrate):
    tp_all, conf_all, pred_cls_all, target_cls_all = [], [], [], []
    n_recal, n_skip = 0, 0
    imgs = sorted(FLARED_IMG.glob("*.jpg"))
    for start in range(0, len(imgs), CHUNK):
        chunk = imgs[start:start + CHUNK]
        results = model.predict([str(p) for p in chunk], imgsz=640, conf=0.0001, iou=0.5,
                                batch=BATCH, verbose=False)
        for r, img_path in zip(results, chunk):
            H, W = r.orig_shape
            gt = read_gt(FLARED_LBL / (img_path.stem + ".txt"), W, H)
            if len(gt):
                target_cls_all.append(gt[:, 0])
            if r.boxes is not None and len(r.boxes):
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy().copy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                if recalibrate:
                    for c, mod in modules.items():
                        idx_all = np.where(clss == c)[0]
                        # v6: only recalibrate boxes the module was trained on
                        # (train crops were real detections at conf >= 0.25).
                        # Junk boxes below 0.25 were never in training — boosting
                        # them scrambled the ranking and cost car -0.134 AP in v5.
                        idx = idx_all[confs[idx_all] >= RECAL_MIN_CONF]
                        n_skip += len(idx_all) - len(idx)
                        n_recal += len(idx)
                        if len(idx) == 0:
                            continue
                        img = r.orig_img  # BGR
                        crops, keep = [], []
                        for i in idx:
                            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                            x1 = max(0, x1); y1 = max(0, y1)
                            x2 = min(W, x2); y2 = min(H, y2)
                            if x2 - x1 < 4 or y2 - y1 < 4:
                                continue
                            patch = cv2.resize(img[y1:y2, x1:x2], (128, 128))
                            crops.append(tf(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))))
                            keep.append(i)
                        if crops:
                            batch = torch.stack(crops).to(DEVICE)
                            with torch.no_grad():
                                drops = mod(batch).cpu().numpy()
                            for i, d in zip(keep, drops):
                                confs[i] = min(1.0, max(0.0, confs[i] + float(d)))
                order = np.argsort(-confs)
                matched = set()
                for i in order:
                    c = clss[i]
                    same = np.where(gt[:, 0] == c)[0] if len(gt) else np.array([], dtype=int)
                    tp = 0
                    if len(same):
                        ious = box_iou(boxes[i:i+1], gt[same, 1:5])[0]
                        best = int(np.argmax(ious))
                        if ious[best] >= 0.5 and int(same[best]) not in matched:
                            matched.add(int(same[best]))
                            tp = 1
                    tp_all.append(tp); conf_all.append(confs[i]); pred_cls_all.append(c)
        del results
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{start + len(chunk)}/{len(imgs)} images done")
    if recalibrate:
        print(f"recalibrated {n_recal} boxes, skipped {n_skip} below {RECAL_MIN_CONF}")
    # v4: ultralytics ap_per_class needs tp 2-D (N, num_iou_thresholds); 1-D crashes
    # on tp.shape[1]. One column = IoU 0.5 only, i.e. mAP50.
    tp = np.array(tp_all).reshape(-1, 1)
    conf = np.array(conf_all)
    pred_cls = np.array(pred_cls_all)
    target_cls = np.concatenate(target_cls_all) if target_cls_all else np.array([])
    # v4: save raw arrays first so a crash in ap_per_class never loses the inference
    tag = "recal_v6" if recalibrate else "plain"
    np.savez(Path(r"C:\flare-yolo") / f"step15_{tag}.npz",
             tp=tp, conf=conf, pred_cls=pred_cls, target_cls=target_cls)
    print(f"saved step15_{tag}.npz")
    return ap_dict(tp, conf, pred_cls, target_cls)

def main():
    print("device:", DEVICE)
    model = YOLO(r"C:\flare-yolo\yolov10m.pt")
    print("== PASS 1: plain (sanity — must match val-4) ==")
    plain_path = Path(r"C:\flare-yolo\step15_plain.npz")
    if plain_path.exists():
        print("loaded step15_plain.npz (Pass 1 inference skipped)")
        ap1 = ap_from_npz(plain_path)
    else:
        ap1 = evaluate(model, {}, recalibrate=False)
    print("plain APs:", {NAMES[c]: round(v, 4) for c, v in sorted(ap1.items())})
    print("== PASS 2: recalibrated ==")
    recal_path = Path(r"C:\flare-yolo\step15_recal_v6.npz")
    if recal_path.exists():
        print("loaded step15_recal_v6.npz (Pass 2 inference skipped)")
        ap2 = ap_from_npz(recal_path)
    else:
        modules = {2: load_module(r"C:\flare-yolo\flare_car.pt"),
                   9: load_module(r"C:\flare-yolo\flare_light.pt")}
        ap2 = evaluate(model, modules, recalibrate=True)
    print(f"{'class':<15}{'plain':>10}{'recal':>10}{'delta':>10}")
    for c in sorted(ap1):
        print(f"{NAMES[c]:<15}{ap1[c]:>10.4f}{ap2[c]:>10.4f}{ap2[c]-ap1[c]:>10.4f}")
    m1 = sum(ap1.values()) / len(ap1); m2 = sum(ap2.values()) / len(ap2)
    print(f"{'mean':<15}{m1:>10.4f}{m2:>10.4f}{m2-m1:>10.4f}")
    print("done")

if __name__ == "__main__":
    main()
