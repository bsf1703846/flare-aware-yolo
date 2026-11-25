# make_figure16.py — Step 16: traffic-light failure figure
# Finds the images where flare hurt traffic lights the most, and draws
# before/after pairs: LEFT = flared image, plain YOLO boxes;
# RIGHT = same flared image, recalibrated boxes (Step 15 v5 method).
# True positives green, false positives blue, ground-truth boxes thin yellow.
# Run on the RTX PC (~10 min for 1200 images). Needs: yolov10m.pt, flare_light.pt,
# C:\flare-yolo\bdd100k\val\images, C:\flare-yolo\flared\images\val,
# C:\flare-yolo\flared\labels\val (COCO-remapped labels, traffic light = 9).
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLEAN_IMG = Path(r"C:\flare-yolo\bdd100k\val\images")
FLARED_IMG = Path(r"C:\flare-yolo\flared\images\val")
FLARED_LBL = Path(r"C:\flare-yolo\flared\labels\val")
OUT = Path(r"C:\flare-yolo\figures")
OUT.mkdir(parents=True, exist_ok=True)
BATCH = 4
N_SCAN = 1200  # images scanned to find the worst cases
CONF = 0.25    # display / scoring threshold (same as Step 13 training regime)
IOU_MATCH = 0.5
TOP_K = 4
TL = 9  # traffic light, COCO index


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


def read_gt_tl(label_path, W, H):
    """Traffic-light GT boxes only, in pixel xyxy."""
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if int(p[0]) != TL:
                continue
            cx, cy, w, h = map(float, p[1:5])
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H,
                          (cx + w / 2) * W, (cy + h / 2) * H])
    return boxes


def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def count_tp(dets, gt_boxes):
    """dets: list of (xyxy, conf) for traffic light. Greedy match by confidence."""
    used = set()
    tp = 0
    for xyxy, _ in sorted(dets, key=lambda d: -d[1]):
        best, best_iou = -1, 0.0
        for j, g in enumerate(gt_boxes):
            if j in used:
                continue
            v = box_iou(xyxy, g)
            if v > best_iou:
                best, best_iou = j, v
        if best >= 0 and best_iou >= IOU_MATCH:
            used.add(best)
            tp += 1
    return tp


def tl_dets(result):
    """Traffic-light detections (xyxy, conf) at CONF threshold, original image coords."""
    out = []
    if result.boxes is not None:
        for box in result.boxes:
            if int(box.cls.item()) != TL:
                continue
            c = float(box.conf.item())
            if c < CONF:
                continue
            out.append((box.xyxy[0].tolist(), c))
    return out


def recal_confs(result, module):
    """Step 15 v5 method: add the module's predicted drop to EVERY traffic-light
    box (all confs), then keep boxes at/above CONF for display."""
    img = result.orig_img  # BGR
    H, W = result.orig_shape
    out = []
    if result.boxes is None:
        return out
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    crops, keep = [], []
    for i in np.where(clss == TL)[0]:
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        patch = cv2.resize(img[y1:y2, x1:x2], (128, 128))
        crops.append(tf(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))))
        keep.append(i)
    drops = np.zeros(len(boxes))
    if crops:
        with torch.no_grad():
            drops[keep] = module(torch.stack(crops).to(DEVICE)).cpu().numpy()
    for i in np.where(clss == TL)[0]:
        c = float(np.clip(confs[i] + drops[i], 0.0, 1.0))
        if c >= CONF:
            out.append((boxes[i].tolist(), c))
    return out


def draw_panel(img_bgr, dets, gt_boxes, title):
    """Draw one panel: TP green, FP blue, GT thin yellow. Returns a PIL image."""
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    used = set()
    for xyxy, _ in sorted(dets, key=lambda x: -x[1]):
        best, best_iou = -1, 0.0
        for j, g in enumerate(gt_boxes):
            if j in used:
                continue
            v = box_iou(xyxy, g)
            if v > best_iou:
                best, best_iou = j, v
        if best >= 0 and best_iou >= IOU_MATCH:
            used.add(best)
            color = (0, 200, 0)  # TP green
        else:
            color = (0, 120, 255)  # FP blue
        d.rectangle(xyxy, outline=color, width=4)
    for g in gt_boxes:
        d.rectangle(g, outline=(255, 220, 0), width=2)  # GT thin yellow
    # title bar
    d.rectangle([0, 0, img.width, 44], fill=(0, 0, 0))
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    d.text((12, 6), title, fill=(255, 255, 255), font=font)
    return img


def main():
    print("device:", DEVICE)
    model = YOLO(r"C:\flare-yolo\yolov10m.pt")
    module = FlareModule()
    module.load_state_dict(torch.load(r"C:\flare-yolo\flare_light.pt",
                                      map_location=DEVICE, weights_only=True))
    module = module.to(DEVICE).eval()

    imgs = sorted(FLARED_IMG.glob("*.jpg"))[:N_SCAN]
    print(f"scanning {len(imgs)} images for worst traffic-light flare loss...")
    scored = []
    for start in range(0, len(imgs), 100):
        chunk = imgs[start:start + 100]
        clean_res = model.predict([str(CLEAN_IMG / p.name) for p in chunk],
                                  imgsz=640, conf=CONF, verbose=False)
        flared_res = model.predict([str(p) for p in chunk],
                                   imgsz=640, conf=CONF, verbose=False)
        for p, rc, rf in zip(chunk, clean_res, flared_res):
            H, W = rf.orig_shape
            gt = read_gt_tl(FLARED_LBL / (p.stem + ".txt"), W, H)
            if not gt:
                continue
            tp_clean = count_tp(tl_dets(rc), gt)
            tp_flared = count_tp(tl_dets(rf), gt)
            lost = tp_clean - tp_flared
            if lost > 0:
                scored.append((lost, len(gt), p.name))
        print(f"{start + len(chunk)}/{len(imgs)} scanned, {len(scored)} with loss so far")
    scored.sort(key=lambda s: (-s[0], -s[1]))
    top = scored[:TOP_K]
    print("worst cases (lost, n_gt, file):")
    for s in top:
        print(" ", s)
    if len(top) < TOP_K:
        print(f"WARNING: only {len(top)} images with traffic-light loss found in scan")

    panels = []
    for lost, n_gt, name in top:
        p = FLARED_IMG / name
        rf = model.predict(str(p), imgsz=640, conf=0.0001, verbose=False)[0]
        H, W = rf.orig_shape
        gt = read_gt_tl(FLARED_LBL / (p.stem + ".txt"), W, H)
        plain = tl_dets(rf)  # conf >= 0.25, original confs
        recal = recal_confs(rf, module)
        img_bgr = cv2.imread(str(p))
        left = draw_panel(img_bgr, plain, gt,
                          f"plain — {len(plain)} boxes (lost {lost} vs clean)")
        right = draw_panel(img_bgr, recal, gt,
                           f"recalibrated — {len(recal)} boxes")
        pair = Image.new("RGB", (left.width * 2 + 12, left.height), (255, 255, 255))
        pair.paste(left, (0, 0))
        pair.paste(right, (left.width + 12, 0))
        pair = pair.resize((1280, int(1280 * pair.height / pair.width)))
        pair.save(OUT / f"fig16_{Path(name).stem}.png")
        print(f"saved fig16_{Path(name).stem}.png")
        panels.append(pair)

    # combined 4-row figure for the summary / repo
    if panels:
        w, h = panels[0].size
        grid = Image.new("RGB", (w, h * len(panels) + 10 * (len(panels) - 1)),
                         (255, 255, 255))
        for i, pan in enumerate(panels):
            grid.paste(pan, (0, i * (h + 10)))
        grid.save(OUT / "fig16_combined.png")
        print("saved fig16_combined.png")
    print("done")


if __name__ == "__main__":
    main()
