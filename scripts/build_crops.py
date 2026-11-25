# build_crops.py — Step 13: build flare-score training data
# For each val image: run YOLOv10m on clean + flared, match boxes by IoU,
# target = conf_clean - conf_flare. Save the FLARED crop (128x128) + target.
# Only car (COCO 2) and traffic light (COCO 9) boxes.
# conf=0.25 here (NOT the val 0.0001): we want real detections, not thousands of junk boxes.
# Cap: 30,000 crops max.
import cv2, numpy as np, csv
from pathlib import Path
from ultralytics import YOLO

model = YOLO(r"C:\flare-yolo\yolov10m.pt")
CLEAN = Path(r"C:\flare-yolo\bdd100k\val\images")
FLARED = Path(r"C:\flare-yolo\flared\images\val")
OUT = Path(r"C:\flare-yolo\crops")
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = {2: "car", 9: "traffic light"}
MAX_CROPS = 30000
CONF = 0.25
IOU_MATCH = 0.5

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / max(area_a + area_b - inter, 1e-9)

def get_boxes(result):
    out = []
    if result.boxes is None:
        return out
    for box in result.boxes:
        c = int(box.cls.item())
        if c not in CLASSES:
            continue
        out.append((c, box.xyxy[0].tolist(), float(box.conf.item())))
    return out

def crop_patch(img, xyxy):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    h, w = img.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    patch = img[y1:y2, x1:x2]
    return cv2.resize(patch, (128, 128))

meta = open(OUT / "meta.csv", "w", newline="")
writer = csv.writer(meta)
writer.writerow(["file", "class", "target", "conf_clean", "conf_flare", "iou"])

n = 0
imgs = sorted(CLEAN.glob("*.jpg"))
for i, img_path in enumerate(imgs):
    if n >= MAX_CROPS:
        break
    flared_img = cv2.imread(str(FLARED / img_path.name))
    if flared_img is None:
        continue
    clean_boxes = get_boxes(model.predict(str(img_path), imgsz=640, conf=CONF, verbose=False)[0])
    flared_boxes = get_boxes(model.predict(str(FLARED / img_path.name), imgsz=640, conf=CONF, verbose=False)[0])
    used = set()
    for c, cbox, cconf in clean_boxes:
        best, best_iou = -1, 0.0
        for j, (fc, fbox, fconf) in enumerate(flared_boxes):
            if fc != c or j in used:
                continue
            v = iou(cbox, fbox)
            if v > best_iou:
                best, best_iou = j, v
        if best >= 0 and best_iou >= IOU_MATCH:
            used.add(best)
            fconf = flared_boxes[best][2]
            fbox = flared_boxes[best][1]
            target = cconf - fconf
            patch = crop_patch(flared_img, fbox)
        else:
            # missed after flare: fully hurt
            fconf, best_iou = 0.0, 0.0
            target = cconf
            patch = crop_patch(flared_img, cbox)
        if patch is None:
            continue
        fname = f"{CLASSES[c]}_{n:06d}.jpg"
        cv2.imwrite(str(OUT / fname), patch)
        writer.writerow([fname, CLASSES[c], round(target, 4), round(cconf, 4), round(fconf, 4), round(best_iou, 4)])
        n += 1
    if (i + 1) % 500 == 0:
        print(f"{i+1}/{len(imgs)} images, crops: {n}")
meta.close()
print("done, crops:", n)
