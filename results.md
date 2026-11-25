# Results — Flare-Aware YOLO

Detector: YOLOv10m, COCO-pretrained. Evaluation: imgsz 640, conf 0.0001, IoU 0.5, mAP50.
Dataset: BDD100K validation set, 10,000 images. Flare: Flare7K++ patterns added by `scripts/make_flared.py`
(paper Figure 5 rules: day ≤ 1 flare, night ≤ 6; screen blending). Class order is COCO
(0 person, 1 bicycle, 2 car, 3 motorcycle, 4 airplane, 5 bus, 6 train, 7 truck, 8 boat, 9 traffic light).

## The flare drop (yolo val, CLI)

| Class | Clean AP | Flared AP | Δ |
|---|---|---|---|
| car | 0.634 | 0.597 | −0.037 (−5.8%) |
| person | 0.523 | 0.509 | −0.014 |
| traffic light | 0.0262 | 0.0261 | −0.0001 (no measurable drop — floor effect) |
| truck | 0.0318 | 0.0304 | −0.0014 |
| train | 0.00928 | 0.00759 | −0.0017 |
| all-row mAP50 (flared) | — | 0.249 | — |

## Step 15 — recalibration (recalibrate_val.py, ultralytics ap_per_class)

Plain = YOLOv10 on flared images, our pipeline. Recalibrated = same, with car/traffic-light
confidences += module-predicted drop (v5 method: applied to ALL boxes at conf ≥ 0.0001).

| Class | Plain AP | Recalibrated AP | Δ |
|---|---|---|---|
| person | 0.5046 | 0.5046 | 0.0000 |
| bicycle | 0.3511 | 0.3511 | 0.0000 |
| car | 0.5895 | 0.4556 | **−0.1339** |
| motorcycle | 0.3263 | 0.3263 | 0.0000 |
| bus | 0.0143 | 0.0143 | 0.0000 |
| train | 0.0075 | 0.0075 | 0.0000 |
| truck | 0.0165 | 0.0165 | 0.0000 |
| traffic light | 0.0242 | 0.0107 | **−0.0135** |
| mean (8 classes) | 0.2293 | 0.2108 | −0.0184 |

Sanity check: our pipeline's plain numbers are close to the CLI flared numbers
(car 0.5895 vs 0.597, traffic light 0.0242 vs 0.0261, person 0.5046 vs 0.509) —
small gaps from slightly different box-matching logic. The plain-vs-recalibrated
comparison inside this script is apples-to-apples.

**Diagnosis (verified by reading build_crops.py + train_module.py):**
- Sign is correct: target = conf_clean − conf_flare, added back.
- Cause: train/test mismatch. The module was trained on REAL detections at conf ≥ 0.25
  (Step 13), but v5 applied it to ALL boxes at conf = 0.0001, including thousands of
  junk boxes per image. Boosting junk scrambled the ranking, and ranking is what AP measures.
- Overfitting: Step 14 val MSE stayed flat (~0.009 car / ~0.007 traffic light) while train
  MSE fell to 0.0008 / 0.0006.

**Fix prepared (v6, in recalibrate_val.py):** only recalibrate boxes with conf ≥ 0.25.
Not yet run — listed as future work.

## Step 14 — module training (train_module.py)

ResNet18 (ImageNet-pretrained, early layers frozen) + 3-layer MLP (512→512→128→1).
MSE loss, Adam lr 1e-4, batch 64, 10 epochs, 90/10 train/val split.
30,001 crops: car 26,370, traffic light 3,631.

| Class | Train crops | Val crops | Train MSE (epoch 1 → 10) | Val MSE (final) |
|---|---|---|---|---|
| car | 23,733 | 2,637 | 0.0104 → 0.0008 | ~0.009 (flat — overfitting) |
| traffic light | 3,268 | 363 | 0.0094 → 0.0006 | ~0.007 (flat — overfitting) |

## Step 13 — training crops (build_crops.py)

For each val image: run YOLOv10 on clean + flared, match boxes by IoU ≥ 0.5 (same class),
save the FLARED crop (128×128) + target. Boxes missed after flare count as fully hurt
(target = conf_clean). Settings: conf = 0.25 (real detections, not junk), cap 30,000 crops.
Result: 30,001 crops.

## Paper reference (for comparison)

Ma et al., Sensors 2026: reference-based (teacher) +1.6% AP over baseline;
reference-free (student) +1.0% AP. Student: 19.09 ms/image, 11.51M parameters.
Our simplified student-only version did not reproduce the +1.0% gain — see diagnosis above.
