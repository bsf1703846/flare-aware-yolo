# make_flared.py — add Flare7K++ patterns to BDD100K val images
# Paper rules (Figure 5): day = bright pixels above threshold -> max 1 flare;
# night -> up to 6 flares. Flare position is random each time.
# Screen blending: result = 1 - (1 - img) * (1 - flare)
import os, random, cv2, numpy as np
from pathlib import Path

SRC = Path(r"C:\flare-yolo\bdd100k\val\images")
DST = Path(r"C:\flare-yolo\flared\images\val")
FLARE_DIR = Path(r"C:\flare-yolo\flare7k\Flare7Kpp\Flare7K\Scattering_Flare")
DST.mkdir(parents=True, exist_ok=True)

flare_files = list(FLARE_DIR.rglob("*.png")) + list(FLARE_DIR.rglob("*.jpg"))
print("flare patterns found:", len(flare_files))

def is_day(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (gray > 200).mean() > 0.02   # many very bright pixels = day

def add_flare(img, flare):
    # screen blending: result = 1 - (1 - img) * (1 - flare)
    flare = cv2.resize(flare, (img.shape[1], img.shape[0]))
    return (255 - (255 - img.astype(np.float32)) * (255 - flare.astype(np.float32)) / 255).astype(np.uint8)

for img_path in sorted(SRC.glob("*.jpg")):
    img = cv2.imread(str(img_path))
    n = 1 if is_day(img) else 6
    out = img.copy()
    for _ in range(n):
        f = cv2.imread(str(random.choice(flare_files)))
        out = add_flare(out, f)
    cv2.imwrite(str(DST / img_path.name), out)
print("done")
