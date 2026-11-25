# train_module.py — Step 14: train the flare-score module (ResNet18 + MLP)
# Trains one model per class: flare_car.pt, then flare_light.pt
# Loss: MSE between predicted score and the target from Step 13 (meta.csv)
# Freeze ResNet18 early layers (conv1/bn1/layer1/layer2), train layer3/layer4 + MLP
# Adam lr 1e-4, batch 64, 10 epochs, 90/10 train/val split
import csv, random
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

CROPS = Path(r"C:\flare-yolo\crops")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 64
EPOCHS = 10
LR = 1e-4

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CropDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        fname, target = self.rows[i]
        img = Image.open(CROPS / fname).convert("RGB")
        return tf(img), torch.tensor(target, dtype=torch.float32)

class FlareModule(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.mlp = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.mlp(self.backbone(x)).squeeze(1)

def train_one(cls_name, rows, out_file):
    random.seed(0)
    random.shuffle(rows)
    n_val = max(1, len(rows) // 10)
    train_rows, val_rows = rows[:-n_val], rows[-n_val:]
    print(f"{cls_name}: train {len(train_rows)}, val {len(val_rows)}")
    train_dl = DataLoader(CropDataset(train_rows), batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(CropDataset(val_rows), batch_size=BATCH)
    model = FlareModule().to(DEVICE)
    for name, p in model.backbone.named_parameters():
        if name.startswith(("conv1", "bn1", "layer1", "layer2")):
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    loss_fn = nn.MSELoss()
    for epoch in range(EPOCHS):
        model.train()
        tot, cnt = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x); cnt += len(x)
        model.eval()
        vtot, vcnt = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                vtot += loss_fn(model(x), y).item() * len(x); vcnt += len(x)
        print(f"{cls_name} epoch {epoch+1}/{EPOCHS} train MSE {tot/cnt:.4f} val MSE {vtot/vcnt:.4f}")
    torch.save(model.state_dict(), out_file)
    print(f"saved {out_file}")

def main():
    print("device:", DEVICE)
    by_class = {"car": [], "traffic light": []}
    with open(CROPS / "meta.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_class[row["class"]].append((row["file"], float(row["target"])))
    for cls_name, rows in by_class.items():
        print(f"{cls_name}: {len(rows)} crops")
    train_one("car", by_class["car"], r"C:\flare-yolo\flare_car.pt")
    if len(by_class["traffic light"]) >= 200:
        train_one("traffic light", by_class["traffic light"], r"C:\flare-yolo\flare_light.pt")
    else:
        print("traffic light crops too few, skipping light model")
    print("done")

if __name__ == "__main__":
    main()
