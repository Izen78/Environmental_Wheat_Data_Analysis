import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
from PIL import Image
from sklearn.model_selection import KFold

# --------------------------
# Dataset
# --------------------------
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
        img = img.resize((256, 256), Image.BILINEAR)

        mask_rgb = Image.open(mask_path).convert("RGB")
        mask_rgb= mask_rgb.resize((256, 256), Image.NEAREST)
        mask = self.rgb_to_id(mask_rgb)
        # mask = Image.fromarray(mask)

        if self.transforms:
            img = self.transforms(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask

# --------------------------
# Training and Validation
# --------------------------
def train_one_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimiser.zero_grad()
        outputs = model(imgs)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)

    return total_loss / total_samples


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)['out']
            loss = criterion(outputs, masks)
            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

    return total_loss / total_samples


EPOCHS = 5
# --------------------------
# Cross-validation training
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = "dataset/images"
    mask_dir = "dataset/masks"
    label_json = "dataset/classes.json"
    all_files = sorted(os.listdir(img_dir))

    valid_exts = [".jpg", ".jpeg", ".png"]  # adjust if you have other formats
    all_files = [
        f for f in sorted(os.listdir(img_dir))
        if os.path.splitext(f)[1].lower() in valid_exts
    ]

    # all_files = all_files[:30]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_files), 1):
        print(f"Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")

        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]

        train_dataset = SegDataset(img_dir, mask_dir, label_json, train_files, transforms=transform)
        val_dataset = SegDataset(img_dir, mask_dir, label_json, val_files, transforms=transform)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        model = fcn_resnet50(weights="DEFAULT")
        with open(label_json) as f:
            classes = json.load(f)
        num_classes = len(classes)
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        model = model.to(device)

        optimiser = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):  # Adjust number of epochs as needed
            train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Fold {fold} Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        torch.save(model.state_dict(), f"fcn_fold{fold}.pth")
        visualize_predictions(model, val_dataset, device, num_images=3)

        # Compute confusion matrix on validation set
        cm = compute_confusion_matrix(model, val_dataset, device, num_classes)
        class_names = [c["name"] for c in classes]  # use names from your JSON
    plot_confusion_matrix(cm, class_names)
        # Compute class-wise IoU
    iou = compute_class_iou(cm)
    plot_class_iou(iou, class_names)
    print("Class-wise IoU:", dict(zip(class_names, iou)))

    miou = np.mean(iou)
    print(f"Overall mIoU: {miou:.4f}")

import matplotlib.pyplot as plt

def visualize_predictions(model, dataset, device, num_images=3):
    model.eval()
    shown = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            if shown >= num_images:
                break

            # --------------------
            # Load original image & mask (before transforms)
            # --------------------
            fname = dataset.file_list[idx]
            img_path = os.path.join(dataset.img_dir, fname)
            mask_path = os.path.join(dataset.mask_dir, fname)

            img_orig = Image.open(img_path).convert("RGB").resize((256, 256), Image.BILINEAR)
            mask_rgb_orig = Image.open(mask_path).convert("RGB").resize((256, 256), Image.NEAREST)

            # Convert mask to class IDs
            mask_class = dataset.rgb_to_id(mask_rgb_orig)

            # --------------------
            # Prepare model input
            # --------------------
            img_input = dataset.transforms(img_orig).unsqueeze(0).to(device)

            # Forward pass
            output = model(img_input)['out']
            pred_mask = output.argmax(1).squeeze(0).cpu().numpy()

            # --------------------
            # Plot
            # --------------------
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_orig)
            axes[0].set_title("Original Image")
            axes[1].imshow(mask_class, cmap="tab20")
            axes[1].set_title("Ground Truth Class Mask")
            axes[2].imshow(pred_mask, cmap="tab20")
            axes[2].set_title("Predicted Mask")

            for ax in axes:
                ax.axis("off")
            plt.savefig(f"fcn_baseline_visual_{idx}.png")
            # plt.show()

            shown += 1
from sklearn.metrics import confusion_matrix
import seaborn as sns

def compute_confusion_matrix(model, dataset, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            fname = dataset.file_list[idx]
            img_path = os.path.join(dataset.img_dir, fname)
            mask_path = os.path.join(dataset.mask_dir, fname)

            # Load image + mask
            img_orig = Image.open(img_path).convert("RGB").resize((256, 256), Image.BILINEAR)
            mask_rgb_orig = Image.open(mask_path).convert("RGB").resize((256, 256), Image.NEAREST)
            mask_class = dataset.rgb_to_id(mask_rgb_orig)

            # Forward pass
            img_input = dataset.transforms(img_orig).unsqueeze(0).to(device)
            output = model(img_input)['out']
            pred_mask = output.argmax(1).squeeze(0).cpu().numpy()

            # Flatten masks and append
            all_labels.extend(mask_class.flatten())
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
    plt.savefig("fcn_baseline_confusion.png")
    # plt.show()

def compute_class_iou(cm):
    intersection = np.diag(cm)  # TP for each class
    ground_truth_set = cm.sum(axis=1)  # TP + FN
    predicted_set = cm.sum(axis=0)     # TP + FP
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / (union + 1e-10)  # avoid division by zero
    return iou

def plot_class_iou(iou, class_names):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=class_names, y=iou)
    plt.ylabel("IoU")
    plt.xlabel("Class")
    plt.title("Class-wise IoU")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.savefig("fcn_baseline_class_iou.png")
    # plt.show()


if __name__ == "__main__":
    main()

