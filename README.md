# Flare-Aware YOLO

Measuring how much lens flare hurts YOLOv10 on real driving images (BDD100K), and testing a small learned recalibration module that re-scores boxes instead of removing flare.

Reimplementation of: Ma et al., "A Reference-Free Lens-Flare-Aware Detector for Autonomous Driving", *Sensors* 2026. DOI: [10.3390/s26082359](https://doi.org/10.3390/s26082359)

## Headline results (mAP50, BDD100K val, 10,000 images, YOLOv10m, 640 px)

| Class | Clean | Flared | Recalibrated | Δ flared → recal. |
|---|---|---|---|---|
| car | 0.634 | 0.597 | 0.456 | −0.134 |
| traffic light | 0.0262 | 0.0261 | 0.0107 | −0.0135 |
| person (not recalibrated) | 0.523 | 0.509 | 0.505 | — |

- **The flare drop is real:** car mAP50 falls 0.634 → 0.597 (−5.8%) when flare is added. That is the main result.
- **Traffic light shows no measurable drop** (0.0262 → 0.0261): a floor effect — at 640 px the model already fails on tiny traffic lights.
- **The recalibration module did not work:** it made car worse (0.590 → 0.456 in our pipeline). The cause is verified: a train/test mismatch (trained on real detections at conf ≥ 0.25, applied to all boxes down to conf 0.0001, including thousands of junk boxes) plus overfitting (val MSE flat ~0.009 while train MSE fell to 0.0008). The paper's +1.0% AP gain did not reproduce in this simplified setup. A negative result with a verified cause is a real result — see [SUMMARY.md](SUMMARY.md) and [results.md](results.md).

## What is in this repo

- `SUMMARY.md` — 2-page project summary (problem, method, results, limitations, future work).
- `results.md` — full per-class tables, training curves, and measurement notes.
- `scripts/make_flared.py` — add Flare7K++ patterns to BDD100K val images (paper Figure 5 rules: day ≤ 1 flare, night ≤ 6; screen blending).
- `scripts/build_crops.py` — build training crops: run YOLOv10 on clean + flared, match boxes by IoU, target = conf_clean − conf_flare. 30,001 crops (26,370 car + 3,631 traffic light).
- `scripts/train_module.py` — train the small module (ResNet18 backbone + 3-layer MLP, MSE loss, Adam 1e-4, batch 64, 10 epochs). Saves `flare_car.pt`, `flare_light.pt`.
- `scripts/recalibrate_val.py` — re-measure flared AP with ultralytics' own `ap_per_class`: Pass 1 plain (sanity), Pass 2 recalibrated. Current version is v6: recalibrate only boxes at conf ≥ 0.25 (the regime the module was trained on) — prepared as the fix for the mismatch above, not yet run.
- `scripts/make_figure16.py` — failure figures: the images where flare hurt traffic lights most, plain vs recalibrated boxes (TP green, FP blue, GT yellow).
- `figures/fig16_combined.jpg` — Step 16 failure figure (plain panels: 0 boxes; recalibrated panels: 10/3/1/15 boxes, mostly false positives on the flare glow).
- `bdd100k_flared.yaml` — dataset config (COCO class order) for the flared val set.

## Honest simplification

The paper trains its student by distillation from a reference-based teacher that sees both the clean and flared image (Smooth L1 loss). We do not build the teacher. Our data is paired by construction (a clean image plus the same image with flare added by us), so we train the student directly on the detector's own confidence drop. That is the paper's "impact of lens flare" idea in its simplest honest form — stated as our simplification, not the paper's exact method.

## Reproduce

Hardware used: AMD Ryzen 9 9950X, 32 GB RAM, NVIDIA RTX 5060 Ti 8 GB. YOLOv10m COCO-pretrained; BDD100K validation set; Flare7K++.

1. `python scripts/make_flared.py` — build the flared val set.
2. `yolo val` on clean and flared sets (see `results.md` for the exact configs) — measures the drop.
3. `python scripts/build_crops.py` — 30,001 training crops.
4. `python scripts/train_module.py` — `flare_car.pt`, `flare_light.pt`.
5. `python scripts/recalibrate_val.py` — plain vs recalibrated AP table.
6. `python scripts/make_figure16.py` — failure figures.

## Author

Hafiz Muhammad Ishaq — project summary dated 25 November 2026.
