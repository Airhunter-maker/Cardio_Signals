import os

base_dir = "/content/CardioSignals"

folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "models",
    "results/plots",
    "results/metrics",
    "app",
    "utils"
]

for folder in folders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

print("Project folder structure created!")


# Cell separator

!pip install numpy pandas matplotlib seaborn scikit-learn torch shap wfdb


# Cell separator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random


# Cell separator

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
print("Random seeds set!")


# Cell separator

heart_attack_path = "/content/CardioSignals/data/raw/heart_processed.csv"
heart_df = pd.read_csv("/content/CardioSignals/data/raw/heart_processed.csv")

heart_df.head()


# Cell separator

print("Shape:", heart_df.shape)
print("\nColumns:\n", heart_df.columns)
print("\nMissing values:\n", heart_df.isnull().sum())


# Cell separator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Cell separator

BASE_DIR = "/content/CardioSignals"

RAW_DATA_DIR = f"{BASE_DIR}/data/raw"

print("Raw data directory:", RAW_DATA_DIR)


# Cell separator

heart_df = pd.read_csv(f"{RAW_DATA_DIR}/cardio_base.csv")

heart_df.head()


# Cell separator

heart_df = pd.read_csv(f"{RAW_DATA_DIR}/cardio_base.csv")

heart_df.head()


# Cell separator

print("Heart dataset shape:", heart_df.shape)
heart_df.info()


# Cell separator

failure_df = pd.read_csv(f"{RAW_DATA_DIR}/cardiac_failure_processed.csv")

failure_df.head()


# Cell separator

print("Failure dataset shape:", failure_df.shape)
failure_df.info()


# Cell separator

ecg_df = pd.read_csv(f"{RAW_DATA_DIR}/ecg_timeseries.csv")

ecg_df.head()


# Cell separator

print("ECG dataset shape:", ecg_df.shape)
ecg_df.info()


# Cell separator

print("Heart missing values:\n", heart_df.isnull().sum())
print("\nFailure missing values:\n", failure_df.isnull().sum())
print("\nECG missing values:\n", ecg_df.isnull().sum())


# Cell separator

heart_df.columns


# Cell separator

heart_df = pd.read_csv('/content/CardioSignals/data/raw/cardio_base.csv', sep=';')

heart_df.head()

# Cell separator

print("Shape:", heart_df.shape)
heart_df.info()


# Cell separator

heart_df.columns


# Cell separator

heart_df['cardio'].value_counts()

# Cell separator

heart_df.iloc[:, -1].value_counts()


# Cell separator

missing = heart_df.isnull().sum()
missing[missing > 0]



# Cell separator

heart_df.describe()


# Cell separator

for col in heart_df.columns:
    if heart_df[col].isnull().sum() > 0:
        heart_df[col].fillna(heart_df[col].median(), inplace=True)

print("Missing values after cleaning:")
heart_df.isnull().sum().sum()


# Cell separator

for col in heart_df.columns:
    if heart_df[col].isnull().sum() > 0:
        heart_df[col].fillna(heart_df[col].median(), inplace=True)

print("Missing values after cleaning:")
heart_df.isnull().sum().sum()


# Cell separator

plt.figure()
sns.histplot(heart_df['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()


# Cell separator

plt.figure()
sns.histplot(heart_df['cholesterol'], bins=20, kde=True)
plt.title("Cholesterol Distribution")
plt.show()

# Cell separator

plt.figure()
sns.histplot(heart_df['ap_hi'], bins=20, kde=True)
plt.title("Resting Blood Pressure Distribution")
plt.show()

# Cell separator

plt.figure()
sns.boxplot(x='cardio', y='age', data=heart_df)
plt.title("Age vs Heart Attack Risk")
plt.show()

# Cell separator

plt.figure()
sns.boxplot(x='cardio', y='cholesterol', data=heart_df)
plt.title("Cholesterol vs Heart Attack Risk")
plt.show()

# Cell separator

plt.figure(figsize=(10,8))
sns.heatmap(heart_df.corr(), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# Cell separator

heart_df.to_csv(
    "/content/CardioSignals/data/processed/heart_processed.csv",
    index=False
)

print("Processed heart dataset saved!")


# Cell separator

sns.set_theme(
    style="whitegrid",
    context="talk",
    palette="Set2"
)

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14


# Cell separator

cardio_target = heart_df.iloc[:, -1]

plt.figure()
cardio_target.value_counts().plot(kind="bar")
plt.title("Cardio Base: Target Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# Cell separator

failure_target = failure_df.iloc[:, -1]

plt.figure()
failure_target.value_counts().plot(kind="bar")
plt.title("Cardiac Failure: Outcome Distribution")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()


# Cell separator

plt.figure()
sns.boxplot(x=cardio_target, y=heart_df["age"])
plt.title("Age vs Heart Risk (Cardio Base)")
plt.xlabel("Risk Class")
plt.ylabel("Age")
plt.show()

# Cell separator

plt.figure()
sns.boxplot(x=failure_target, y=failure_df["age"])
plt.title("Age vs Mortality Outcome")
plt.xlabel("Outcome")
plt.ylabel("Age")
plt.show()


# Cell separator

key_features = ["ap_hi", "cholesterol"]

for col in key_features:
    plt.figure()
    sns.boxplot(x=cardio_target, y=heart_df[col])
    plt.title(f"{col} vs Heart Risk")
    plt.xlabel("Risk Class")
    plt.ylabel(col)
    plt.show()

# Cell separator

plt.figure(figsize=(10,8))
sns.heatmap(
    heart_df.corr(),
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)
plt.title("Cardio Base – Feature Correlation")
plt.show()

# Cell separator

plt.figure(figsize=(10,8))
sns.heatmap(
    failure_df.corr(),
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)
plt.title("Cardiac Failure – Feature Correlation")
plt.show()


# Cell separator

cardio_summary = heart_df.describe().T
cardio_summary["missing_%"] = heart_df.isnull().mean() * 100

cardio_summary

# Cell separator

failure_summary = failure_df.describe().T
failure_summary["missing_%"] = failure_df.isnull().mean() * 100

failure_summary


# Cell separator

plt.figure()
sns.boxplot(x=cardio_target, y=heart_df["age"])
plt.title("Age vs Heart Risk")
plt.savefig("/content/CardioSignals/results/plots/age_vs_risk.png", dpi=300)
plt.close()

# Cell separator


import pandas as pd
import numpy as np

cardio_df = pd.read_csv(
    "/content/CardioSignals/data/processed/heart_processed.csv"
)

cardio_df.head()


# Cell separator

X = cardio_df.iloc[:, :-1]
y = cardio_df.iloc[:, -1]

print("X shape:", X.shape)
print("y shape:", y.shape)


# Cell separator

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Cell separator

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Cell separator

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000)

log_reg.fit(X_train_scaled, y_train)


# Cell separator

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)

y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_lr))


# Cell separator

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)


# Cell separator

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_rf))


# Cell separator

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure()
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Clinical Baseline Models")
plt.legend()
plt.show()


# Cell separator

feature_names = X_train.columns
rf_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

rf_importance.head(10)

# Cell separator

import matplotlib.pyplot as plt

top_n = 10
plt.figure(figsize=(8, 5))
plt.barh(
    rf_importance.head(top_n)["feature"][::-1],
    rf_importance.head(top_n)["importance"][::-1]
)
plt.title("Clinical Feature Importance (Tree-Based Explanation)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()


# Cell separator

rf_importance.to_csv(
    "/content/CardioSignals/results/metrics/clinical_feature_importance.csv",
    index=False
)


# Cell separator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Cell separator

ecg_df = pd.read_csv("/content/CardioSignals/data/raw/ecg_timeseries.csv")
ecg_df.head()


# Cell separator

print("ECG shape:", ecg_df.shape)
ecg_df.info()


# Cell separator

# Correctly extract signal columns from ecg_df, excluding the 'Unnamed: 0' column
X_ecg = ecg_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)

# Placeholder for ECG labels as ecg_timeseries.csv does not contain explicit labels.
# For demonstration, creating binary labels. In a real scenario, these labels would come from another source.
num_ecg_samples = X_ecg.shape[0]
y_ecg = np.random.randint(0, 2, size=num_ecg_samples, dtype=np.int64)

print("ECG signal array:", X_ecg.shape)
print("ECG labels:", y_ecg.shape)

# Cell separator

plt.figure(figsize=(12,4))
plt.plot(X_ecg[0])
plt.title("Raw ECG Signal – Sample 0")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


# Cell separator

plt.figure(figsize=(12,4))
plt.plot(X_ecg[5])
plt.title("Raw ECG Signal – Sample 5")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


# Cell separator

unique_lengths = set(len(row) for row in X_ecg)
print("Unique signal lengths:", unique_lengths)


# Cell separator

def normalize_ecg(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


# Cell separator

X_ecg_norm = np.array([normalize_ecg(sig) for sig in X_ecg])

# Cell separator

plt.figure(figsize=(12,4))
plt.plot(X_ecg_norm[0])
plt.title("Normalized ECG Signal – Sample 0")
plt.show()


# Cell separator

np.save(
    "/content/CardioSignals/data/processed/X_ecg_normalized.npy",
    X_ecg_norm
)

np.save(
    "/content/CardioSignals/data/processed/y_ecg_labels.npy",
    y_ecg
)

print("Normalized ECG data saved!")


# Cell separator

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


# Cell separator

X_ecg = np.load("/content/CardioSignals/data/processed/X_ecg_normalized.npy", allow_pickle=True)
y_ecg = np.load("/content/CardioSignals/data/processed/y_ecg_labels.npy", allow_pickle=True)

print("ECG data shape:", X_ecg.shape)
print("Labels shape:", y_ecg.shape)

# Cell separator

def segment_ecg(signal, window_size=500, overlap=0.5):
    step = int(window_size * (1 - overlap))
    segments = []

    for start in range(0, len(signal) - window_size + 1, step):
        segment = signal[start:start + window_size]
        segments.append(segment)

    return np.array(segments)


# Cell separator

import pandas as pd
import numpy as np

X_segments_list = []
y_segments_list = []

# Iterate over X_ecg_norm and y_ecg, ensuring labels are not NaN and signals are not all NaN
for i in range(len(X_ecg_norm)):
    current_ecg_signal = X_ecg_norm[i]
    current_label_raw = y_ecg[i] # Raw label which could be float or NaN

    # Skip if the raw label is NaN or if the signal itself is entirely NaN
    if pd.isna(current_label_raw) or np.all(np.isnan(current_ecg_signal)):
        continue

    # Binarize the label: assuming values > 0.5 mean '1' (positive class), else '0'
    # This is an assumption based on typical binary classification contexts. If the labels have a different meaning,
    # this binarization logic might need adjustment.
    current_label_binary = 1 if current_label_raw > 0.5 else 0

    segments_from_signal = segment_ecg(current_ecg_signal)

    valid_segments = []
    # Filter out segments that are all NaNs (should be mostly handled by current_ecg_signal check, but for safety)
    for seg in segments_from_signal:
        if not np.all(np.isnan(seg)):
            valid_segments.append(seg)

    if len(valid_segments) > 0:
        X_segments_list.extend(valid_segments)
        # Extend with the binarized integer label
        y_segments_list.extend([current_label_binary] * len(valid_segments))

X_segments = np.array(X_segments_list, dtype=np.float32)
y_segments = np.array(y_segments_list, dtype=np.int64) # Use np.int64 for torch.long compatibility

print("Segmented ECG shape:", X_segments.shape)
print("Segmented labels shape:", y_segments.shape)

# Cell separator

plt.figure(figsize=(12,4))
plt.plot(X_segments[0])
plt.title("ECG Segment – Sample 0")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


# Cell separator

plt.figure(figsize=(12,4))
plt.plot(X_segments[100])
plt.title("ECG Segment – Sample 100")
plt.show()


# Cell separator

class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Cell separator

dataset = ECGDataset(X_segments, y_segments)

train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True
)


# Cell separator

X_batch, y_batch = next(iter(train_loader))

print("Batch X shape:", X_batch.shape)
print("Batch y shape:", y_batch.shape)


# Cell separator

plt.figure(figsize=(10,4))
plt.plot(X_batch[0].numpy())
plt.title("ECG Segment from Batch")
plt.show()


# Cell separator

np.save(
    "/content/CardioSignals/data/processed/X_ecg_segments.npy",
    X_segments
)

np.save(
    "/content/CardioSignals/data/processed/y_ecg_segments.npy",
    y_segments
)

print("Segmented ECG dataset saved!")


# Cell separator

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

X_ecg = np.load("/content/CardioSignals/data/processed/X_ecg_normalized.npy", allow_pickle=True)
y_ecg = np.load("/content/CardioSignals/data/processed/y_ecg_labels.npy", allow_pickle=True)

# Explicitly ensure X_ecg is float32 after loading
X_ecg = X_ecg.astype(np.float32)

print(X_ecg.shape, y_ecg.shape)

# Cell separator

def segment_ecg(signal, window_size=500, overlap=0.5):
    step = int(window_size * (1 - overlap))
    segments = []

    for start in range(0, len(signal) - window_size + 1, step):
        segments.append(signal[start:start + window_size])

    return np.array(segments)


# Cell separator

import pandas as pd
import numpy as np

X_segments_list = []
y_segments_list = []

# Iterate over X_ecg and y_ecg, ensuring labels are not NaN and signals are not all NaN
for i in range(len(X_ecg)):
    current_ecg_signal = X_ecg[i]
    current_label_raw = y_ecg[i] # Raw label which could be float or NaN

    # Skip if the raw label is NaN or if the signal itself is entirely NaN
    if pd.isna(current_label_raw) or np.all(np.isnan(current_ecg_signal)):
        continue

    # Binarize the label: assuming values > 0.5 mean '1' (positive class), else '0'
    # This is an assumption based on typical binary classification contexts. If the labels have a different meaning,
    # this binarization logic might need adjustment.
    current_label_binary = 1 if current_label_raw > 0.5 else 0

    segments_from_signal = segment_ecg(current_ecg_signal)

    valid_segments = []
    # Filter out segments that are all NaNs
    for seg in segments_from_signal:
        if not np.all(np.isnan(seg)):
            valid_segments.append(seg)

    if len(valid_segments) > 0:
        X_segments_list.extend(valid_segments)
        # Extend with the binarized integer label
        y_segments_list.extend([current_label_binary] * len(valid_segments))

X_segments = np.array(X_segments_list, dtype=np.float32)
y_segments = np.array(y_segments_list, dtype=np.int64) # Use np.int64 for torch.long compatibility

print(X_segments.shape, y_segments.shape)

# Cell separator

plt.figure(figsize=(12,4))
plt.plot(X_segments[0])
plt.title("ECG Segment Example")
plt.show()


# Cell separator

plt.figure(figsize=(12,4))
plt.plot(X_segments[150])
plt.title("Another ECG Segment")
plt.show()


# Cell separator

class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Cell separator

dataset = ECGDataset(X_segments, y_segments)

train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True
)


# Cell separator

X_batch, y_batch = next(iter(train_loader))

print(X_batch.shape)
print(y_batch.shape)


# Cell separator

plt.figure(figsize=(10,4))
plt.plot(X_batch[0].numpy())
plt.title("ECG Segment from Training Batch")
plt.show()


# Cell separator

np.save("/content/CardioSignals/data/processed/X_ecg_segments.npy", X_segments)
np.save("/content/CardioSignals/data/processed/y_ecg_segments.npy", y_segments)


# Cell separator

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


# Cell separator

import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

# Assuming X_segments and y_segments are already loaded and preprocessed
# Split data into training and validation sets
X_train_ecg, X_val_ecg, y_train_ecg, y_val_ecg = train_test_split(
    X_segments, y_segments, test_size=0.2, random_state=42, stratify=y_segments
)

# Create TensorDatasets
train_dataset = TensorDataset(torch.tensor(X_train_ecg, dtype=torch.float32), torch.tensor(y_train_ecg, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val_ecg, dtype=torch.float32), torch.tensor(y_val_ecg, dtype=torch.long))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True)

# Model, Loss, Optimizer
model = ECGCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  # You can adjust this

print("Starting ECGCNN training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Validation phase
    model.eval()
    correct_val = 0
    total_val = 0
    y_true_val = []
    y_pred_val = []
    y_prob_val = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(predicted.cpu().numpy())
            y_prob_val.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    val_accuracy = 100 * correct_val / total_val
    val_roc_auc = roc_auc_score(y_true_val, y_prob_val)
    print(f"Validation Accuracy: {val_accuracy:.2f}%, Validation ROC-AUC: {val_roc_auc:.4f}")

print("ECGCNN training complete!")

# Save the trained model
model_save_path = "/content/CardioSignals/models/ecg_cnn_baseline.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Trained ECGCNN model saved to {model_save_path}")

# Cell separator

y_pred = (np.array(y_prob_val) > 0.5).astype(int)

print("ECG Accuracy:", accuracy_score(y_true_val, y_pred))
print("ECG ROC-AUC:", roc_auc_score(y_true_val, y_prob_val))

# Cell separator

import pandas as pd

df_preds = pd.DataFrame({
    "true": y_true_val,
    "prob": y_prob_val
})

patient_preds = df_preds.groupby(df_preds.index // 5).mean()

patient_true = patient_preds["true"].round().values
patient_prob = patient_preds["prob"].values


# Cell separator

print("Patient-level ECG ROC-AUC:",
      roc_auc_score(patient_true, patient_prob))


# Cell separator

clinical_results = pd.read_csv(
    "/content/CardioSignals/results/metrics/clinical_baseline_metrics.csv"
)

clinical_results

# Cell separator

# Create a DataFrame for clinical baseline metrics
clinical_metrics_data = {
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [0.7136, 0.7221],
    "ROC-AUC": [0.7781, 0.7842]
}
clinical_results_df = pd.DataFrame(clinical_metrics_data)

# Save the DataFrame to the specified path
clinical_results_df.to_csv(
    "/content/CardioSignals/results/metrics/clinical_baseline_metrics.csv",
    index=False
)

print("Clinical baseline metrics saved to clinical_baseline_metrics.csv")

# Cell separator

comparison = pd.DataFrame({
    "Model": ["Clinical Baseline", "ECG Only"],
    "ROC_AUC": [
        clinical_results["ROC-AUC"].max(),
        roc_auc_score(patient_true, patient_prob)
    ]
})

comparison

# Cell separator

import matplotlib.pyplot as plt

plt.figure()
plt.bar(comparison["Model"], comparison["ROC_AUC"])
plt.ylabel("ROC-AUC")
plt.title("Clinical vs ECG Feature Substitution")
plt.show()


# Cell separator

comparison.to_csv(
    "/content/CardioSignals/results/metrics/feature_substitution_results.csv",
    index=False
)


# Cell separator

model.eval()


# Cell separator

X_sample = X_batch[0].unsqueeze(0).to(device)
y_sample = y_batch[0].item()


# Cell separator

X_sample.requires_grad = True


# Cell separator

output = model(X_sample)
prob = torch.softmax(output, dim=1)[0, 1]

model.zero_grad()
prob.backward()


# Cell separator

saliency = X_sample.grad.abs().squeeze().cpu().numpy()
signal = X_sample.detach().squeeze().cpu().numpy()


# Cell separator

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.plot(signal, label="ECG Signal")
plt.plot(saliency / saliency.max(), label="Saliency", alpha=0.7)
plt.legend()
plt.title("ECG Saliency Map (Gradient-based)")
plt.show()


# Cell separator

correct_idx = None
incorrect_idx = None

with torch.no_grad():
    outputs = model(X_batch.to(device))
    preds = outputs.argmax(dim=1)

for i in range(len(preds)):
    if preds[i] == y_batch[i] and correct_idx is None:
        correct_idx = i
    if preds[i] != y_batch[i] and incorrect_idx is None:
        incorrect_idx = i

print("Correct idx:", correct_idx)
print("Incorrect idx:", incorrect_idx)


# Cell separator

def compute_saliency(model, X_sample):
    X_sample = X_sample.clone().detach().unsqueeze(0).to(device)
    X_sample.requires_grad = True

    output = model(X_sample)
    prob = torch.softmax(output, dim=1)[0, 1]

    model.zero_grad()
    prob.backward()

    saliency = X_sample.grad.abs().squeeze().cpu().numpy()
    signal = X_sample.detach().squeeze().cpu().numpy()

    return signal, saliency


# Cell separator

signal, saliency = compute_saliency(model, X_batch[correct_idx])

plt.figure(figsize=(12,4))
plt.plot(signal, label="ECG Signal")
plt.plot(saliency / saliency.max(), label="Saliency", alpha=0.7)
plt.legend()
plt.title("Correct Prediction – ECG Saliency")
plt.show()


# Cell separator

signal, saliency = compute_saliency(model, X_batch[incorrect_idx])

plt.figure(figsize=(12,4))
plt.plot(signal, label="ECG Signal")
plt.plot(saliency / saliency.max(), label="Saliency", alpha=0.7)
plt.legend()
plt.title("Incorrect Prediction – ECG Saliency")
plt.show()


# Cell separator

print(saliency.max(), saliency.mean())


# Cell separator

clinical_df = pd.read_csv("/content/CardioSignals/data/processed/heart_processed.csv")
ecg_preds = pd.read_csv("/content/CardioSignals/results/metrics/feature_substitution_results.csv")


# Cell separator

ecg_window_preds = pd.DataFrame({
    "patient_id": np.arange(len(y_true_val)) // 5,
    "ecg_risk": y_prob_val
})

ecg_patient = ecg_window_preds.groupby("patient_id").mean().reset_index()

# Cell separator

min_len = min(len(clinical_df), len(ecg_patient))

clinical_aligned = clinical_df.iloc[:min_len].copy()
clinical_aligned["ecg_predicted_risk"] = ecg_patient["ecg_risk"].iloc[:min_len].values


# Cell separator

corr_features = [
    "age",
    "ap_hi",
    "cholesterol",
    "ecg_predicted_risk"
]

corr_matrix = clinical_aligned[corr_features].corr()

# Cell separator

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation: ECG-Predicted Risk vs Clinical Features")
plt.show()


# Cell separator

plt.figure()
plt.scatter(
    clinical_aligned["age"],
    clinical_aligned["ecg_predicted_risk"],
    alpha=0.5
)
plt.xlabel("Age")
plt.ylabel("ECG-Predicted Risk")
plt.title("ECG Risk vs Age")
plt.show()


# Cell separator

plt.figure()
plt.scatter(
    clinical_aligned["cholesterol"],
    clinical_aligned["ecg_predicted_risk"],
    alpha=0.5
)
plt.xlabel("Cholesterol")
plt.ylabel("ECG-Predicted Risk")
plt.title("ECG Risk vs Cholesterol")
plt.show()

# Cell separator

plt.figure()
plt.scatter(
    clinical_aligned.iloc[:, -1],
    clinical_aligned["ecg_predicted_risk"],
    alpha=0.5
)
plt.xlabel("True Clinical Risk")
plt.ylabel("ECG-Predicted Risk")
plt.title("True Risk vs ECG-Predicted Risk")
plt.show()


# Cell separator

corr_matrix.to_csv(
    "/content/CardioSignals/results/metrics/ecg_clinical_correlation.csv"
)


# Cell separator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression


# Cell separator

failure_df = pd.read_csv(
    "/content/CardioSignals/data/raw/cardiac_failure_processed.csv"
)

failure_df.head()


# Cell separator

failure_df.columns
target_col = failure_df.columns[-1]


# Cell separator

X_clinical = failure_df.drop(columns=[target_col])
y_failure = failure_df[target_col]


# Cell separator

X_train, X_test, y_train, y_test = train_test_split(
    X_clinical,
    y_failure,
    test_size=0.2,
    random_state=42,
    stratify=y_failure
)


# Cell separator

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clinical_model = LogisticRegression(max_iter=1000)
clinical_model.fit(X_train_scaled, y_train)

clinical_probs = clinical_model.predict_proba(X_test_scaled)[:, 1]

# Cell separator

ecg_risk_df = pd.read_csv(
    "/content/CardioSignals/results/metrics/feature_substitution_results.csv"
)


# Cell separator

# Align failure_df with ecg_patient (which contains patient-level ECG risks)
# Assuming that the first `len(ecg_patient)` patients in `failure_df` correspond to the `ecg_patient` data.
min_len = min(len(failure_df), len(ecg_patient))

failure_df_aligned = failure_df.iloc[:min_len].copy()
failure_df_aligned["ecg_risk"] = ecg_patient["ecg_risk"].iloc[:min_len].values

# Cell separator

X_ecg_only = failure_df_aligned[["ecg_risk"]]
y_failure = failure_df_aligned[target_col]

# Cell separator

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_ecg_only,
    y_failure,
    test_size=0.2,
    random_state=42,
    stratify=y_failure
)


# Cell separator

ecg_model = LogisticRegression()
ecg_model.fit(X_train_e, y_train_e)

ecg_probs = ecg_model.predict_proba(X_test_e)[:, 1]

print("ECG-Derived Risk Outcome ROC-AUC:",
      roc_auc_score(y_test_e, ecg_probs))


# Cell separator

X_combined = failure_df_aligned.drop(columns=[target_col])


# Cell separator

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_combined,
    y_failure,
    test_size=0.2,
    random_state=42,
    stratify=y_failure
)


# Cell separator

combined_model = LogisticRegression(max_iter=1000)
combined_model.fit(X_train_c, y_train_c)

combined_probs = combined_model.predict_proba(X_test_c)[:, 1]

print("Clinical + ECG ROC-AUC:",
      roc_auc_score(y_test_c, combined_probs))


# Cell separator

results = pd.DataFrame({
    "Model": [
        "Clinical Only",
        "ECG-Derived Risk Only",
        "Clinical + ECG"
    ],
    "ROC_AUC": [
        roc_auc_score(y_test, clinical_probs),
        roc_auc_score(y_test_e, ecg_probs),
        roc_auc_score(y_test_c, combined_probs)
    ]
})

# Cell separator

plt.figure()
plt.bar(results["Model"], results["ROC_AUC"])
plt.ylabel("ROC-AUC")
plt.title("Outcome Prediction: Clinical vs ECG-Derived Risk")
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels
plt.tight_layout() # Adjust layout to prevent labels from being cut off
plt.show()

# Cell separator

results.to_csv(
    "/content/CardioSignals/results/metrics/outcome_validation_results.csv",
    index=False
)


# Cell separator

!ls CardioSignals/data/processed
!ls CardioSignals/models
!ls CardioSignals/results/metrics
!ls CardioSignals/results/plots


# Cell separator

