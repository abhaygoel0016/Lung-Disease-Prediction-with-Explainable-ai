

import os, json, random, time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# ─────────────────────────── CONFIG ───────────────────────────
DATA_CSV    = "data/Data_Entry_2017.csv"
IMG_DIR     = "data/images"
MODEL_DIR   = "models"
IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_EPOCHS  = 15
LR          = 1e-4
SEED        = 42

# NIH 14 pathology labels
ALL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

os.makedirs(MODEL_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────── DATASET ──────────────────────────
class ChestXRayDataset(Dataset):
    def __init__(self, df, img_dir, labels, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image Index"])
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Multi-label binary vector
        label_str = row["Finding Labels"]
        target = torch.zeros(len(self.labels), dtype=torch.float32)
        for i, lbl in enumerate(self.labels):
            if lbl in label_str:
                target[i] = 1.0

        return image, target, row["Image Index"]


def get_transforms(mode="train"):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ─────────────────────────── MODEL ────────────────────────────
class CheXNet(nn.Module):
    """DenseNet-121 with a custom multi-label classification head."""
    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet121(weights=weights)
        # Replace the final classifier
        in_features = densenet.classifier.in_features
        densenet.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self.model = densenet

    def forward(self, x):
        return self.model(x)          # Raw logits; use BCEWithLogitsLoss


# ─────────────────────────── TRAINING ─────────────────────────
def compute_auc(outputs, targets):
    """Mean AUC over all classes (ignores classes with no positive samples)."""
    outputs = torch.sigmoid(outputs).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    aucs = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            aucs.append(roc_auc_score(targets[:, i], outputs[:, i]))
    return np.mean(aucs) if aucs else 0.0


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0
    for imgs, targets, _ in loader:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            outputs = model(imgs)
            loss    = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_outputs, all_targets = [], []
    for imgs, targets, _ in loader:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        outputs = model(imgs)
        total_loss += criterion(outputs, targets).item()
        all_outputs.append(outputs)
        all_targets.append(targets)
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    auc = compute_auc(all_outputs, all_targets)
    return total_loss / len(loader), auc


# ─────────────────────────── MAIN ─────────────────────────────
def main():
    # ── Load CSV ──
    df = pd.read_csv(DATA_CSV)
    # Keep only rows where image file actually exists (handy during dev)
    df = df[df["Image Index"].apply(lambda f: os.path.exists(os.path.join(IMG_DIR, f)))]
    print(f"Total usable images: {len(df)}")

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=SEED)
    val_df, test_df  = train_test_split(val_df, test_size=0.5, random_state=SEED)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # ── DataLoaders ──
    train_ds = ChestXRayDataset(train_df, IMG_DIR, ALL_LABELS, get_transforms("train"))
    val_ds   = ChestXRayDataset(val_df,   IMG_DIR, ALL_LABELS, get_transforms("val"))
    test_ds  = ChestXRayDataset(test_df,  IMG_DIR, ALL_LABELS, get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ──
    model     = CheXNet(num_classes=len(ALL_LABELS)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    # Class-frequency-aware positive weights (helps with imbalance)
    pos_counts = train_df["Finding Labels"].apply(
        lambda s: pd.Series({l: int(l in s) for l in ALL_LABELS})
    ).sum()
    neg_counts = len(train_df) - pos_counts
    pos_weight = torch.tensor((neg_counts / pos_counts.clip(lower=1)).values,
                               dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    # ── Training loop ──
    best_auc   = 0
    history    = []
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0        = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_auc = evaluate(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Time: {elapsed:.1f}s")

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_auc": val_auc})

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_auc":     val_auc,
                "labels":      ALL_LABELS,
            }, os.path.join(MODEL_DIR, "best_model.pth"))
            print(f"  ✓ Best model saved (AUC={best_auc:.4f})")

    # ── Final test evaluation ──
# NEW
    checkpoint = torch.load(os.path.join(MODEL_DIR, "best_model.pth"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_auc = evaluate(model, test_loader, criterion)
    print(f"\nTest AUC: {test_auc:.4f}  (Total training: {(time.time()-start_time)/60:.1f} min)")

    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(ALL_LABELS, f, indent=2)

    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(MODEL_DIR, "training_history.csv"), index=False)
    print("Training complete. Artifacts saved to models/")


if __name__ == "__main__":
    main()
