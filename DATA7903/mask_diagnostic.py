# mask_diagnostic.py
import os
import json
import numpy as np
from PIL import Image
from collections import Counter

IMG_MASK_DIR = "dataset/masks"   # change if needed
CLASSES_JSON = "dataset/classes.json"
SAMPLE_N = None  # set to integer to sample N masks (None = all)

def load_classes(path):
    with open(path) as f:
        labels = json.load(f)
    cls_colors = {tuple(l["color"]): (l["id"], l["name"]) for l in labels}
    return cls_colors, labels

def unique_colors_in_mask(mask_path):
    im = Image.open(mask_path).convert("RGB")
    arr = np.array(im)
    colors = np.unique(arr.reshape(-1, 3), axis=0)
    # return list of tuples
    return [tuple(c.tolist()) for c in colors]

def main():
    cls_colors, labels = load_classes(CLASSES_JSON)
    print("Declared classes in JSON:")
    for c in labels:
        print(f"  id={c['id']}, name={c['name']}, color={c['color']}")

    mask_files = sorted([f for f in os.listdir(IMG_MASK_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if SAMPLE_N:
        mask_files = mask_files[:SAMPLE_N]

    observed_counter = Counter()
    files_with_unexpected = []

    for mf in mask_files:
        path = os.path.join(IMG_MASK_DIR, mf)
        colors = unique_colors_in_mask(path)
        for c in colors:
            observed_counter[c] += 1
        # if any observed color not in classes.json: record file
        unexpected = [c for c in colors if tuple(c) not in cls_colors]
        if unexpected:
            files_with_unexpected.append((mf, unexpected))

    print("\nUnique observed colors (sampled counts):")
    for col, cnt in observed_counter.most_common():
        print(f"  {col}  - seen in {cnt} masks")

    if files_with_unexpected:
        print(f"\nFiles containing unexpected colours: {len(files_with_unexpected)} examples (showing up to 10):")
        for i, (fn, cols) in enumerate(files_with_unexpected[:10], 1):
            print(f" {i}. {fn} -> unexpected colors: {cols}")
    else:
        print("\nAll observed mask colours are present in classes.json ✅")

if __name__ == "__main__":
    main()
