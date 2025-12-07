import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from collections import Counter

# ===============================
# CONFIGURATION
# ===============================
IMG_DIR = "dataset/images"
MASK_DIR = "dataset/masks"
LABEL_JSON = "dataset/classes.json"  # optional if you have class definitions
VALID_EXTS = [".jpg", ".jpeg", ".png"]

# ===============================
# 1. CHECK FILE CORRESPONDENCE
# ===============================
image_files = sorted([f for f in os.listdir(IMG_DIR) if os.path.splitext(f)[1].lower() in VALID_EXTS])
mask_files = sorted([f for f in os.listdir(MASK_DIR) if os.path.splitext(f)[1].lower() in VALID_EXTS])

print(f"Total Images: {len(image_files)}")
print(f"Total Masks: {len(mask_files)}")

missing_masks = set(image_files) - set(mask_files)
missing_images = set(mask_files) - set(image_files)

if missing_masks:
    print("⚠️ Missing masks for:", list(missing_masks)[:10])
if missing_images:
    print("⚠️ Missing images for:", list(missing_images)[:10])
else:
    print("✅ All image–mask pairs matched.")

# ===============================
# 2. ANALYSE IMAGE DIMENSIONS
# ===============================
dims = []
for f in image_files[:200]:  # sample up to 200 for speed
    img = Image.open(os.path.join(IMG_DIR, f))
    dims.append(img.size)  # (width, height)
dims = np.array(dims)

plt.figure(figsize=(8,6))
plt.scatter(dims[:,0], dims[:,1], alpha=0.6)
plt.xlabel("Width (px)")
plt.ylabel("Height (px)")
plt.title("Image Dimensions Distribution")
plt.grid(True)
plt.show()

aspect_ratios = dims[:,0] / dims[:,1]
sns.histplot(aspect_ratios, bins=20, kde=True)
plt.title("Aspect Ratio Distribution (Width / Height)")
plt.show()

# ===============================
# 3. VISUALIZE RANDOM IMAGE + MASK
# ===============================
def show_samples(n=3):
    sample_files = random.sample(image_files, n)
    for fname in sample_files:
        img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
        mask = Image.open(os.path.join(MASK_DIR, fname)).convert("RGB")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title(f"Original Image ({fname})")
        axes[1].imshow(mask)
        axes[1].set_title("Mask (RGB)")

        # Overlay visualization
        overlay = Image.blend(img, mask, alpha=0.4)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (Image + Mask)")
        for ax in axes:
            ax.axis("off")
        plt.show()

show_samples(3)

# ===============================
# 4. CLASS DISTRIBUTION (if masks use class colors)
# ===============================
if os.path.exists(LABEL_JSON):
    with open(LABEL_JSON) as f:
        label_info = json.load(f)
    colour_to_id = {tuple(l["color"]): l["id"] for l in label_info}
    id_to_name = {l["id"]: l["name"] for l in label_info}
else:
    colour_to_id = None
    print("⚠️ No label JSON found, skipping class mapping.")

def rgb_to_id(mask_rgb, colour_to_id):
    mask_array = np.array(mask_rgb)
    class_mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)
    for clr, idx in colour_to_id.items():
        matches = np.all(mask_array == clr, axis=-1)
        class_mask[matches] = idx
    return class_mask

class_counts = Counter()
for fname in random.sample(mask_files, min(50, len(mask_files))):  # sample subset
    mask = Image.open(os.path.join(MASK_DIR, fname)).convert("RGB")
    if colour_to_id:
        mask = rgb_to_id(mask, colour_to_id)
    arr = np.array(mask).flatten()
    unique, counts = np.unique(arr, return_counts=True)
    for u, c in zip(unique, counts):
        class_counts[u] += c

if colour_to_id:
    labels, freqs = zip(*[(id_to_name[k], v) for k, v in sorted(class_counts.items())])
else:
    labels, freqs = zip(*sorted(class_counts.items()))

plt.figure(figsize=(10,5))
sns.barplot(x=labels, y=freqs)
plt.title("Class Pixel Frequency Distribution")
plt.xlabel("Class")
plt.ylabel("Pixel Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===============================
# 5. COLOR HISTOGRAM ANALYSIS
# ===============================
def plot_color_hist(image_path):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    plt.figure(figsize=(6,4))
    for i, color in enumerate(["Red", "Green", "Blue"]):
        plt.hist(arr[:,:,i].ravel(), bins=50, alpha=0.5, label=color)
    plt.legend()
    plt.title(f"Color Histogram: {os.path.basename(image_path)}")
    plt.show()

plot_color_hist(os.path.join(IMG_DIR, random.choice(image_files)))
