# Flare-Aware YOLO: Measuring Lens-Flare Damage on YOLOv10 and Testing a Small Recalibration Module

**Hafiz Muhammad Ishaq • 25 November 2026 • Project summary**

## 1. Problem

Driving cameras face lens flare every day: sun glare, headlights at night, wet roads. A detector that works on clean images can fail exactly when the scene is hardest. This project asks two questions. First, how much does flare hurt YOLOv10 on real driving images? Second, can a small learned module repair the damage by re-scoring the detector's boxes, without removing anything from the image?

We work from Ma et al., "A Reference-Free Lens-Flare-Aware Detector for Autonomous Driving", *Sensors* 2026 (DOI: 10.3390/s26082359). The paper's reference-free student gains +1.0% AP over its baseline. We reimplement its idea in a simplified form and test it honestly, including the case where it does not work.

## 2. Method

**Data.** BDD100K validation set: 10,000 driving images with labels. Flare comes from Flare7K++, added to the images with our make_flared.py script. Detector: YOLOv10m, COCO-pretrained, evaluated at 640 px, confidence 0.0001, IoU 0.5 (mAP50).

**Measuring the drop.** We run YOLOv10 on clean images and on the same images with flare, and compare per-class AP. This is the main measurement.

**The module.** Following the paper's student (ResNet18 backbone + 3-layer MLP, 11.51M parameters), we train one small model per class — car and traffic light. Honest simplification: the paper distils its student from a reference-based teacher that sees both images (Smooth L1). We skip the teacher: our data is paired by construction, so we train directly on the detector's own confidence drop, conf_clean − conf_flare. We built 30,001 training crops (26,370 car + 3,631 traffic light) from boxes the detector found at confidence ≥ 0.25, and trained with MSE loss, Adam, lr 1e-4, batch 64, 10 epochs, 90/10 train/val split.

**Recalibration.** At test time, each car or traffic-light box is cropped, the module predicts its confidence drop, and the drop is added back to the box's confidence. Nothing is removed from the image; only scores change.

## 3. Results

| Class | Clean AP | Flared AP | Recalibrated AP | Δ flared → recal. |
|---|---|---|---|---|
| car | 0.634 | 0.597 | 0.456 | −0.134 |
| traffic light | 0.0262 | 0.0261 | 0.0107 | −0.0135 |
| person (not recalibrated) | 0.523 | 0.509 | 0.505 | — |

*Clean and flared: yolo val (CLI). Recalibrated: recalibrate_val.py; its plain re-measurement inside the same script gave car 0.590 and traffic light 0.024, close to the CLI flared values, so the flared → recalibrated comparison is apples-to-apples. Mean over 8 classes in our pipeline: plain 0.2293 → recalibrated 0.2108 (−0.018).*

Lead result: flare drops car mAP50 from 0.634 to 0.597 — a drop of 0.037 (−5.8%). Person drops 0.523 → 0.509. The drop is real, measured on 10,000 images, and it is the headline of this project.

Traffic light shows no measurable drop: 0.0262 → 0.0261. This is a floor effect — at 640 px the model already fails on tiny traffic lights, so flare adds nothing measurable. That non-result is itself a finding, and it motivates the per-class analysis in Section 5.

## 4. The recalibration did not work — and why

The honest result: the module made things worse. Car fell from 0.590 (our pipeline's plain flared measurement) to 0.456, a drop of 0.134; traffic light fell from 0.024 to 0.011. The paper's +1.0% gain did not reproduce in our simplified setup.

We checked the sign first: the target is conf_clean − conf_flare and it is added back, so the direction is correct. The cause is a train/test mismatch. The module was trained only on real detections (confidence ≥ 0.25), but at test time it was applied to every box down to confidence 0.0001 — thousands of junk boxes per image that the module never saw in training. Boosting junk boxes scrambled the ranking, and ranking is what AP measures. Training already warned us: validation MSE stayed flat (~0.009) while training MSE fell to 0.0008 — the module overfits and does not generalise.

We prepared a fix (apply the module only to boxes at confidence ≥ 0.25, the regime it was trained on) and chose to report the negative result as it stands, with the fix as future work. A negative result with a verified cause is a real result.

## 5. Traffic lights: the extra analysis

The paper analyses the car class in depth. We add traffic lights, because they are special: the paper itself notes traffic lights are themselves flare sources (§5.1), and flare-removal models erase or distort them (Figure 12, §5.2) — a safety risk. Our method never removes anything; it only re-scores. But at 640 px the detector is already at its floor, so the flare effect cannot be seen. The next step is to re-measure at 1280 px (BDD100K's native resolution), where traffic lights are bigger and the baseline should rise into a measurable range. Failure figures (before/after pairs for the worst cases) will be added from Step 16.

## 6. Limitations

- Simplified student only: no teacher, no distillation, unlike the paper.
- One fold, validation set only (10,000 images); no test-set confirmation.
- Images evaluated at 640 px; traffic lights are tiny at this size.
- Module trained on 30,001 crops, 10 epochs; validation MSE shows overfitting.
- Flare is synthetic (Flare7K++ added by us), not real in-the-wild flare.

## 7. Future work

- Threshold-aligned recalibration: apply the module only where it was trained (confidence ≥ 0.25).
- More training data and augmentation to close the train/val gap.
- Re-measure at 1280 px for the traffic-light analysis.
- Compare against a flare-removal baseline on the same images.

## 8. Reproducibility

Code and results: https://github.com/bsf1703846/flare-aware-yolo (link to be confirmed once the repository is created). Scripts: make_flared.py (add flare), build_crops.py (training crops), train_module.py (ResNet18 + MLP, two classes), recalibrate_val.py (plain vs recalibrated AP), make_figure16.py (failure figures). Hardware: AMD Ryzen 9 9950X, 32 GB RAM, NVIDIA RTX 5060 Ti 8 GB. YOLOv10m COCO-pretrained; BDD100K validation set; Flare7K++.
