# ===============================
# objective2.py
# ===============================

# --- 0. Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# --- 1. Load data ---
df = pd.read_csv('merged_final_dataset3.csv')
print("Data shape:", df.shape)

# Exclude frost damage >= 5
df = df[df['Frost damage'] < 5]
print("Filtered dataset shape (<5):", df.shape)

label_col = 'Frost damage'
print("Label distribution:\n", df[label_col].value_counts())

# --- 2. Preprocessing ---
exclude_cols = ['Name','TrialCode','SiteDescription','General Comments (DO NOT USE)','Frost damage']
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Separate numeric/categorical
num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

num_cols_nonempty = [c for c in num_cols if df[c].notna().any()]
cat_cols_nonempty = [c for c in cat_cols if df[c].notna().any()]

# Impute numeric
num_imputer = SimpleImputer(strategy='median')
df_num = pd.DataFrame(num_imputer.fit_transform(df[num_cols_nonempty]), columns=num_cols_nonempty)

# Impute categorical + one-hot
df_cat = pd.DataFrame()
if len(cat_cols_nonempty) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df_cat_imp = pd.DataFrame(cat_imputer.fit_transform(df[cat_cols_nonempty]), columns=cat_cols_nonempty)
    df_cat = pd.get_dummies(df_cat_imp, drop_first=True)

# Combine
X_full = pd.concat([df_num, df_cat], axis=1)
y_full = df[label_col].astype(int)
print("Final feature matrix shape:", X_full.shape)

# --- 3. Train/test/validation splits ---
# Logistic regression 70/30
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_full, y_full, test_size=0.30, stratify=y_full, random_state=42
)

# XGBoost + NN 60/20/20
X_temp, X_test_xgbnn, y_temp, y_test_xgbnn = train_test_split(
    X_full, y_full, test_size=0.20, stratify=y_full, random_state=42
)
X_train_xgbnn, X_val_xgbnn, y_train_xgbnn, y_val_xgbnn = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# --- 4. Scaling ---
scaler = StandardScaler()
def scale_numeric(X, numeric_cols, scaler=None, fit=True):
    X_scaled = X.copy()
    if fit:
        X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        X_scaled[numeric_cols] = scaler.transform(X[numeric_cols])
    return X_scaled

X_train_log_scaled = scale_numeric(X_train_log, num_cols_nonempty, scaler, fit=True)
X_test_log_scaled  = scale_numeric(X_test_log, num_cols_nonempty, scaler, fit=False)
X_train_xgbnn_scaled = scale_numeric(X_train_xgbnn, num_cols_nonempty, scaler, fit=True)
X_val_xgbnn_scaled   = scale_numeric(X_val_xgbnn, num_cols_nonempty, scaler, fit=False)
X_test_xgbnn_scaled  = scale_numeric(X_test_xgbnn, num_cols_nonempty, scaler, fit=False)

# Fill NaNs and ensure float32
X_train_xgbnn_scaled = X_train_xgbnn_scaled.astype(np.float32).fillna(0)
X_val_xgbnn_scaled   = X_val_xgbnn_scaled.astype(np.float32).fillna(0)
X_test_xgbnn_scaled  = X_test_xgbnn_scaled.astype(np.float32).fillna(0)

# --- 5. Class weights ---
classes = np.unique(y_full)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_full)
class_weight_dict = {int(c): w for c, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)

# =========================
# Model 1: Logistic Regression
# =========================
log_clf = LogisticRegression(
    multi_class='multinomial',
    solver='saga',
    max_iter=2000,
    class_weight=class_weight_dict,
    n_jobs=-1,
    random_state=42
)
log_clf.fit(X_train_log_scaled, y_train_log)
y_pred_log = log_clf.predict(X_test_log_scaled)
y_proba_log = log_clf.predict_proba(X_test_log_scaled)
print("Logistic Regression test accuracy:", accuracy_score(y_test_log, y_pred_log))
print(classification_report(y_test_log, y_pred_log))

out_log = X_test_log.copy()
out_log['true'] = y_test_log.values
out_log['pred'] = y_pred_log
proba_cols = [f'prob_{c}' for c in log_clf.classes_]
out_log = pd.concat([out_log, pd.DataFrame(y_proba_log, columns=proba_cols, index=out_log.index)], axis=1)
out_log.to_csv('Objective2/powerbi_logistic_predictions_all.csv', index=False)

# =========================
# Model 2: XGBoost
# =========================
# Shift labels to 0-based for XGBoost
y_train_xgbnn_0 = y_train_xgbnn - y_train_xgbnn.min()
y_val_xgbnn_0   = y_val_xgbnn - y_val_xgbnn.min()
y_test_xgbnn_0  = y_test_xgbnn - y_test_xgbnn.min()

dtrain = xgb.DMatrix(X_train_xgbnn_scaled, label=y_train_xgbnn_0)
dval   = xgb.DMatrix(X_val_xgbnn_scaled, label=y_val_xgbnn_0)
dtest  = xgb.DMatrix(X_test_xgbnn_scaled, label=y_test_xgbnn_0)

params = {
    'objective': 'multi:softprob',
    'num_class': len(classes),
    'eval_metric': 'mlogloss',
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}
watchlist = [(dtrain,'train'), (dval,'val')]
bst = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist,
                early_stopping_rounds=30, verbose_eval=50)

y_proba_xgb = bst.predict(dtest)
y_pred_xgb = np.argmax(y_proba_xgb, axis=1) + y_full.min()
print("XGBoost test accuracy:", accuracy_score(y_test_xgbnn, y_pred_xgb))
print(classification_report(y_test_xgbnn, y_pred_xgb))

out_xgb = X_test_xgbnn.copy()
out_xgb['true'] = y_test_xgbnn.values
out_xgb['pred'] = y_pred_xgb
proba_cols = [f'prob_{int(i + y_full.min())}' for i in range(len(classes))]
out_xgb = pd.concat([out_xgb, pd.DataFrame(y_proba_xgb, columns=proba_cols, index=out_xgb.index)], axis=1)
out_xgb.to_csv('Objective2/powerbi_xgb_predictions_all.csv', index=False)
bst.save_model('Objective2/xgb_frost_all.model')

# =========================
# Model 3: Neural Network
# =========================
X_train_tensor = torch.tensor(X_train_xgbnn_scaled.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_xgbnn_0.values, dtype=torch.long)
X_val_tensor   = torch.tensor(X_val_xgbnn_scaled.values, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val_xgbnn_0.values, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test_xgbnn_scaled.values, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test_xgbnn_0.values, dtype=torch.long)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
val_ds   = TensorDataset(X_val_tensor, y_val_tensor)
test_ds  = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

class FrostNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FrostNet(X_train_xgbnn_scaled.shape[1], len(classes)).to(device)

class_weight_tensor = torch.tensor(list(class_weight_dict.values()), dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Train loss {total_loss/len(train_loader):.4f}")

model.eval()
y_test_preds, y_test_probs = [], []
with torch.no_grad():
    for Xb, _ in test_loader:
        Xb = Xb.to(device)
        outputs = model(Xb)
        y_test_probs.append(F.softmax(outputs, dim=1).cpu().numpy())
        y_test_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())

y_test_probs = np.vstack(y_test_probs)
y_test_preds = np.concatenate(y_test_preds) + y_full.min()
y_test_true = y_test_xgbnn.values

print("NN test accuracy:", accuracy_score(y_test_true, y_test_preds))
print(classification_report(y_test_true, y_test_preds))

out_nn = X_test_xgbnn.copy()
out_nn['true'] = y_test_true
out_nn['pred'] = y_test_preds
proba_cols = [f'prob_{int(i + y_full.min())}' for i in range(len(classes))]
out_nn = pd.concat([out_nn, pd.DataFrame(y_test_probs, columns=proba_cols, index=out_nn.index)], axis=1)
out_nn.to_csv('Objective2/powerbi_nn_predictions_all.csv', index=False)
torch.save(model.state_dict(), 'Objective2/nn_frost_all.pt')
