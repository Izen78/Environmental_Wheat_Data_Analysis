import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from sklearn.model_selection import KFold

import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation, SegformerConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    print("Using Cuda")


class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, label_json, file_list, transforms=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.mask_transform = mask_transform
        self.file_list = file_list

        with open(label_json) as f:
            labels = json.load(f)
        self.colour_to_id = {tuple(l["color"]): l["id"] for l in labels}

    def rgb_to_id(self, mask_rgb):
        #mask_array = np.array(mask_rgb)
        #class_mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)
        #for clr, i in self.colour_to_id.items():
        #    matches = np.all(mask_array==clr, axis=1)
        #    class_mask[matches]=i
        #return class_mask

        mask_array = np.array(mask_rgb)

        if mask_array.ndim == 3 and mask_array.shape[0]==3:
            mask_array = np.transpose(mask_array, (1,2,0))

        h, w, _ = mask_array.shape
        class_mask = np.zeros((h,w), dtype=np.uint8)

        for color, class_id in self.colour_to_id.items():
            matches = np.all(mask_array == color, axis=-1)
            class_mask[matches] = class_id

        return class_mask
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        fname = self.file_list[index]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        img = Image.open(img_path).convert("RGB")
        mask_rgb = Image.open(mask_path).convert("RGB")
        mask = self.rgb_to_id(mask_rgb)
        mask = Image.fromarray(mask)

        if self.transforms:
            img = self.transforms(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask

train_transforms = T.Compose([
    T.Resize((512, 512)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

val_transforms = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

mask_transform = T.Compose([
    T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST)
])

def get_segformer(num_classes, pretrained=True):
    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512",
                num_labels=num_classes,
                ignore_mismatched_sizes=True)
    else:
        config = SegformerConfig(num_labels=num_classes)
        model = SegformerForSemanticSegmentation(config)

    return model

def train_one_epoch(model, loader, optimiser, criterion, scaler):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimiser.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(pixel_values=imgs)
            # SegFormer logits are [B, C, H/4, W/4] typically; upsample to mask size
            logits = outputs.logits
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            optimiser.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        outputs = model(pixel_values=imgs)
        logits = outputs.logits
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        loss = criterion(logits, masks)

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


def main():
    img_root = "dataset/images"
    mask_root = "dataset/masks"
    labels_path = "dataset/classes.json"

    with open(labels_path) as f:
        labels = json.load(f)
    num_classes = len(labels)

    # all_images = sorted(os.listdir(img_root))
    valid_exts = [".jpg", ".jpeg", ".png"]  # add other extensions if needed
    all_images = [
        f for f in sorted(os.listdir(img_root))
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    # all_images = all_images[:20]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    EPOCHS = 5 # was 10
    BATCH_SIZE = 4
    LR = 6e-5            # typical for transformer fine-tuning
    WD = 0.01            # AdamW weight decay
    NUM_WORKERS = 4

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images), 1):
        print(f"--------------- FOLD {fold} ---------------")
        train_files = [all_images[i] for i in train_idx]
        val_files   = [all_images[i] for i in val_idx]
        print(f" Train: {len(train_files)}, Val: {len(val_files)}")

        train_dataset = SegDataset(img_root, mask_root, labels_path, train_files,
                                   transforms=train_transforms, mask_transform=mask_transform)
        val_dataset   = SegDataset(img_root, mask_root, labels_path, val_files,
                                   transforms=val_transforms, mask_transform=mask_transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"), drop_last=True)
        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))

        model = get_segformer(num_classes=num_classes, pretrained=True).to(device)

        # AdamW is recommended for transformers
        optimiser = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

        # CE loss: labels already class indices [0..num_classes-1]
        criterion = nn.CrossEntropyLoss()  # add ignore_index=X if you have unlabeled pixels

        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimiser, criterion, scaler)
            val_loss = validate(model, val_loader, criterion)
            print(f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Optional: save the fold model
        os.makedirs("checkpoints_segformer", exist_ok=True)
        # torch.save(model.state_dict(), f"checkpoints_segformer/segformer_fold{fold}.pth")

    all_labels_total = []
    all_preds_total = []

    # load labels
    with open(labels_path) as f:
        labels = json.load(f)
    num_classes = len(labels)
    class_names = [l["name"] for l in labels]

    # inside fold loop
    for idx in range(len(val_dataset)):
        img, mask = val_dataset[idx]
        img_input = img.unsqueeze(0).to(device)

        outputs = model(pixel_values=img_input)
        logits = F.interpolate(outputs.logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        pred_mask = logits.argmax(1).squeeze(0).cpu().numpy()

        all_labels_total.extend(mask.cpu().numpy().flatten())
        all_preds_total.extend(pred_mask.flatten())

    # after all folds
    all_labels_total = np.array(all_labels_total)
    all_preds_total = np.array(all_preds_total)

    cm = confusion_matrix(all_labels_total, all_preds_total, labels=list(range(num_classes)))
    plot_confusion_matrix(cm, class_names)

    intersection = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        pred_c = all_preds_total == c
        true_c = all_labels_total == c
        intersection[c] = np.logical_and(pred_c, true_c).sum()
        union[c] = np.logical_or(pred_c, true_c).sum()
    class_iou = intersection / (union + 1e-10)
    plot_class_iou(class_iou, class_names)


    visualize_predictions(model, val_dataset, device, num_images=3)

        # # Compute confusion matrix on validation set
        # cm = compute_confusion_matrix(model, val_dataset, device, num_classes)
        # class_names = [l["name"] for l in labels]  # use names from your JSON
        # plot_confusion_matrix(cm, class_names)

        # # Compute class-wise IoU
        # class_iou = compute_class_iou(model, val_dataset, device, num_classes)
        # plot_class_iou(class_iou, class_names)


import matplotlib.pyplot as plt

@torch.no_grad()
def visualize_predictions(model, dataset, device, num_images=3):
    model.eval()
    shown = 0

    for idx in range(len(dataset)):
        if shown >= num_images:
            break

        img, mask = dataset[idx]
        img_input = img.unsqueeze(0).to(device)

        # Forward pass
        outputs = model(pixel_values=img_input)
        logits = outputs.logits
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        pred_mask = logits.argmax(1).squeeze(0).cpu().numpy()

        img_np = img.permute(1,2,0).cpu().numpy()  # C,H,W -> H,W,C

        # --------------------
        # Plot
        # --------------------
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[1].imshow(mask.cpu().numpy(), cmap="tab20")
        axes[1].set_title("Ground Truth Mask")
        axes[2].imshow(pred_mask, cmap="tab20")
        axes[2].set_title("Predicted Mask")

        for ax in axes:
            ax.axis("off")
        plt.savefig(f"segformer_baseline_visual_{idx}.png")
        # plt.show()
        # plt.savefig(f"segformer_visual_{idx}.png")  # optionally save
        shown += 1

from sklearn.metrics import confusion_matrix
import seaborn as sns

@torch.no_grad()
def compute_confusion_matrix(model, dataset, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []

    for idx in range(len(dataset)):
        img, mask = dataset[idx]
        img_input = img.unsqueeze(0).to(device)

        outputs = model(pixel_values=img_input)
        logits = outputs.logits
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        pred_mask = logits.argmax(1).squeeze(0).cpu().numpy()

        all_labels.extend(mask.cpu().numpy().flatten())
        all_preds.extend(pred_mask.flatten())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return cm

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("segformer_baseline_confusion.png")
    # plt.show()

@torch.no_grad()
def compute_class_iou(model, dataset, device, num_classes):
    """
    Compute class-wise Intersection over Union (IoU) for segmentation.
    Returns a list of IoUs per class.
    """
    model.eval()
    intersection = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    for idx in range(len(dataset)):
        img, mask = dataset[idx]
        img_input = img.unsqueeze(0).to(device)

        outputs = model(pixel_values=img_input)
        logits = outputs.logits
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        pred_mask = logits.argmax(1).squeeze(0).cpu().numpy()
        true_mask = mask.cpu().numpy()

        for c in range(num_classes):
            pred_c = (pred_mask == c)
            true_c = (true_mask == c)
            intersection[c] += np.logical_and(pred_c, true_c).sum()
            union[c] += np.logical_or(pred_c, true_c).sum()

    class_iou = intersection / (union + 1e-10)  # add epsilon to avoid divide by zero
    return class_iou

def plot_class_iou(class_iou, class_names):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=class_iou)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("IoU")
    plt.title("Class-wise IoU")
    plt.ylim(0, 1)
    plt.savefig("segformer_baseline_class_iou.png")
    # plt.show()

if __name__ == "__main__":
    print("segformer baseline")
    main()
