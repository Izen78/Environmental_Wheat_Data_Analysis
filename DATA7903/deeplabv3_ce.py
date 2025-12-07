import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random

from sklearn.model_selection import KFold

import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    print("Using Cuda")
print("deeplabv3_baseline_v2")

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

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.0, path="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last improvement.
            verbose (bool): Print updates when saving a new best model.
            delta (float): Minimum improvement in loss to be considered.
            path (str): Where to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss improves."""
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} → {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
class MorphologicalCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, boundary_weight=1.0, ignore_index=255, stem_class=2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index
        self.stem_class = stem_class  # e.g., 2 if stem = class 2

        # Laplacian kernel
        kernel = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
        self.register_buffer("laplacian_kernel", kernel)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        # One-hot encode targets: [B, C, H, W]
        num_classes = logits.shape[1]
        targets_oh = F.one_hot(
            targets.clamp(min=0), num_classes=num_classes
        ).permute(0, 3, 1, 2).float()

        # Expand kernel for depthwise conv
        kernel = self.laplacian_kernel.repeat(num_classes, 1, 1, 1)  # [C,1,3,3]

        # Detect edges per class
        edges = F.conv2d(targets_oh, kernel.to(targets_oh.dtype), padding=1, groups=num_classes).abs()

        # Compute probs
        probs = F.softmax(logits, dim=1)

        # Boundary emphasis
        boundary_loss = (edges * -torch.log(probs + 1e-8)).mean()

        # Extra emphasis on stems
        stem_edges = edges[:, self.stem_class, :, :]
        stem_probs = probs[:, self.stem_class, :, :]
        stem_boundary_loss = (stem_edges * -torch.log(stem_probs + 1e-8)).mean()

        total_loss = ce_loss + self.boundary_weight * (boundary_loss + 2.0 * stem_boundary_loss)
        return total_loss

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

def get_deeplab_model(num_classes):
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

# Training Loop
def train_one_epoch(model, loader, optimiser, criterion):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimiser.zero_grad()
        outputs = model(imgs)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

# Validation Loop
    T.Resize((512, 512)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])


val_transforms = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

mask_transform = T.Compose([
    T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST)
])

def get_deeplab_model(num_classes):
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

# Training Loop
def train_one_epoch(model, loader, optimiser, criterion):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimiser.zero_grad()
        outputs = model(imgs)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

# Validation Loop
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)["out"]
            loss = criterion(outputs, masks)
            total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


# Cross Validation
img_dir = "dataset/images"
# all_images = sorted(os.listdir(img_dir))
valid_exts = [".jpg", ".jpeg", ".png"]  # adjust if you have other formats
all_images = [
    f for f in sorted(os.listdir(img_dir))
    if os.path.splitext(f)[1].lower() in valid_exts
]

# random.seed(42)
all_images = random.sample(all_images, 20)

kf = KFold(n_splits=2, shuffle=True, random_state=42) # was 2

with open("dataset/classes.json") as f:
    labels = json.load(f)
num_classes = len(labels)

EPOCHS = 1
BATCH_SIZE = 4
LR = 1e-4
labels = "dataset/classes.json"
for fold, (train_idx, val_idx) in enumerate(kf.split(all_images), 1):
    print(f"---------------FOLD {fold}---------------")
    train_files = [all_images[i] for i in train_idx]
    val_files = [all_images[i] for i in val_idx]
    print(f" Train: {len(train_files)}, Val: {len(val_files)}")

    train_dataset = SegDataset("dataset/images", "dataset/masks", labels, train_files, transforms=train_transforms, mask_transform=mask_transform)
    val_dataset = SegDataset("dataset/images", "dataset/masks", labels, val_files, transforms=val_transforms, mask_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_deeplab_model(num_classes).to(device)
    # criterion = nn.CrossEntropyLoss()

    class_weights = torch.tensor([1.0, 3.0, 3.0,2.0], device=device)  # example: background=1, leaf=3, stem=3
    criterion = MorphologicalCrossEntropyLoss(weight=class_weights, boundary_weight=2.0, ignore_index=255)
    optimiser = optim.Adam(model.parameters(), lr=LR)

    # 🔹 Early stopping setup (per fold)
    checkpoint_path = f"checkpoints_deeplabv3/best_fold{fold}.pth"
    os.makedirs("checkpoints_deeplabv3", exist_ok=True)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=checkpoint_path)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion)
        val_loss = validate(model, val_loader, criterion)
        print(f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

# 🔹 Load the best model before evaluation
model.load_state_dict(torch.load(checkpoint_path))
import matplotlib.pyplot as plt
def visualize_predictions(model, dataset, device, num_images=3):
    """
    Visualize a few predictions from the model.
    Shows original image, ground truth mask, and predicted mask.
    """
    model.eval()
    images_shown = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            if images_shown >= num_images:
                break

            # Load original image and mask without transforms
            fname = dataset.file_list[idx]
            img_path = os.path.join(dataset.img_dir, fname)
            mask_path = os.path.join(dataset.mask_dir, fname)

            img_orig = Image.open(img_path).convert("RGB")
            mask_orig = Image.open(mask_path).convert("RGB")

            # Transform image for model input
            img_input = dataset.transforms(img_orig).unsqueeze(0).to(device)
                        # Forward pass
            output = model(img_input)["out"]
            pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()

            # Convert ground truth mask to class ids
            mask_gt = dataset.rgb_to_id(mask_orig)

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_orig)
            axes[0].set_title("Original Image")
            axes[1].imshow(mask_gt, cmap="tab20")
            axes[1].set_title("Ground Truth Mask")
            axes[2].imshow(pred_mask, cmap="tab20")
            axes[2].set_title("Predicted Mask")

            for ax in axes:
                ax.axis("off")
            plt.savefig(f"deeplabv3_baseline_visual_{idx}.png")
            # plt.show()

            images_shown += 1
# After training your model on one fold:
visualize_predictions(model, val_dataset, device, num_images=3)

def compute_confusion_matrix(model, dataset, device, num_classes):
    """
    Compute a confusion matrix over the entire dataset.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            fname = dataset.file_list[idx]
            img_path = os.path.join(dataset.img_dir, fname)
            mask_path = os.path.join(dataset.mask_dir, fname)

            img_orig = Image.open(img_path).convert("RGB")
            mask_orig = Image.open(mask_path).convert("RGB")

            # Model input
            img_input = dataset.transforms(img_orig).unsqueeze(0).to(device)
            output = model(img_input)["out"]
            pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()

            # Ground truth mask with resizing
            mask_gt = dataset.rgb_to_id(mask_orig)
            mask_gt = Image.fromarray(mask_gt)
            if dataset.mask_transform:
                mask_gt = dataset.mask_transform(mask_gt)
            mask_gt = np.array(mask_gt)

            # Ensure shapes match
            if mask_gt.shape != pred_mask.shape:
                print(f"Shape mismatch for {fname}: pred {pred_mask.shape}, gt {mask_gt.shape}")
                continue

            all_preds.append(pred_mask.flatten())
            all_labels.append(mask_gt.flatten())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return cm


def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10,8))
    ious = []
    for i in range(len(cm)):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - cm[i, i]
        if union == 0:
            iou = float("nan")  # no samples of this class
        else:
            iou = intersection / union
        ious.append(iou)
    miou = np.nanmean(ious)
    return ious, miou


def plot_iou(ious, labels, miou):
    """
    Plot per-class IoU and show mean IoU.
    """
    class_ids = [l["id"] for l in labels]
    plt.figure(figsize=(10,6))
    plt.bar(class_ids, ious)
    plt.xticks(class_ids)
    plt.ylim(0, 1)
    plt.xlabel("Class ID")
    plt.ylabel("IoU")
    plt.title(f"Per-Class IoU (mIoU = {miou:.3f})")
    plt.savefig("deeplabv3_baseline_miou.png")
    # plt.show()

cm = compute_confusion_matrix(model, val_dataset, device, num_classes)

with open("dataset/classes.json") as f:
    labels = json.load(f)

plot_confusion_matrix(cm, labels)

ious, miou = compute_iou_per_class(cm)
print("Per-class IoU:", ious)
print("Mean IoU:", miou)

plot_iou(ious, labels, miou)

