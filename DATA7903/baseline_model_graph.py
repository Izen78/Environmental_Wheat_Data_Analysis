import re
import pandas as pd
import matplotlib.pyplot as plt

# Load log file
with open("training_log.txt", "r", encoding="utf-8") as f:
    log_text = f.read()

# Split by fold
fold_splits = re.split(r"-{5,}\s*FOLD\s*\d+\s*-{5,}", log_text, flags=re.IGNORECASE)
fold_splits = [s.strip() for s in fold_splits if "Epoch" in s]

# Regex pattern for DeepLabV3 / Segformer
pattern = r"Epoch\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]\s*-\s*Train Loss:\s*([0-9.eE+-]+)\s*-\s*Val Loss:\s*([0-9.eE+-]+)"

fold_dfs = []
for i, fold_log in enumerate(fold_splits, 1):
    matches = re.findall(pattern, fold_log, flags=re.IGNORECASE)
    if not matches:
        print(f"⚠️ No matches found in fold {i}")
        continue
    epochs = [int(m[0]) for m in matches]
    train_losses = [float(m[2]) for m in matches]
    val_losses = [float(m[3]) for m in matches]

    df = pd.DataFrame({
        "epoch": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "fold": i
    })
    fold_dfs.append(df)

if not fold_dfs:
    raise RuntimeError("No epochs parsed — check if log format matches regex.")

# Combine all folds
all_folds = pd.concat(fold_dfs)

# Plot
plt.figure(figsize=(10,6))
for fold, df in all_folds.groupby("fold"):
    plt.plot(df["epoch"], df["train_loss"], linestyle="--", label=f"Train Fold {fold}")
    plt.plot(df["epoch"], df["val_loss"], label=f"Val Fold {fold}")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DeepLabV3 Baseline + Early Stopping Loss Curves")
plt.legend()
plt.grid(True)
plt.show()
