# Fix target mapping by stripping whitespace and normalizing case

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Paths
DATA_PATH = "Meta.csv"
OUT_DIR = Path("/mnt/data/task4_outputs")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

# Peek unique values in loan_status
unique_status = sorted(df["loan_status"].unique().tolist())

# Clean string columns: strip spaces
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str).str.strip()

# Drop ID
if "loan_id" in df.columns:
    df = df.drop(columns=["loan_id"])

# Encode categoricals
for col in ["education", "self_employed"]:
    if col in df.columns and df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Map target safely (case-insensitive)
df["loan_status_clean"] = df["loan_status"].str.lower().map({"approved": 1, "rejected": 0})
# If any still NaN, try original casing without spaces
mask_missing = df["loan_status_clean"].isna()
if mask_missing.any():
    df.loc[mask_missing, "loan_status_clean"] = df.loc[mask_missing, "loan_status"].map({"Approved": 1, "Rejected": 0})

# Verify no NaN in target
assert df["loan_status_clean"].notna().all(), f"Unmapped loan_status values: {df.loc[df['loan_status_clean'].isna(), 'loan_status'].unique()}"

# Features/target
X = df.drop(columns=["loan_status", "loan_status_clean"])
y = df["loan_status_clean"].astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Helper
def train_eval(clf, name, X_tr, y_tr, X_te, y_te, save_prefix):
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_te)[:, 1]
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_te)
        s_min, s_max = scores.min(), scores.max()
        y_proba = (scores - s_min) / (s_max - s_min + 1e-9)
    else:
        y_proba = y_pred.astype(float)

    report = classification_report(y_te, y_pred, output_dict=True)
    cm = confusion_matrix(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    auc_val = roc_auc_score(y_te, y_proba)

    # Confusion matrix figure
    plt.figure()
    im = plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix – {name}")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Rejected", "Approved"])
    plt.yticks(tick_marks, ["Rejected", "Approved"])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_path = FIG_DIR / f"{save_prefix}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=160, bbox_inches="tight")
    plt.close()

    # ROC figure
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {name}")
    plt.legend(loc="lower right")
    roc_path = FIG_DIR / f"{save_prefix}_roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=160, bbox_inches="tight")
    plt.close()

    metrics_row = {
        "Model": name,
        "Precision_0": report["0"]["precision"],
        "Recall_0": report["0"]["recall"],
        "F1_0": report["0"]["f1-score"],
        "Precision_1": report["1"]["precision"],
        "Recall_1": report["1"]["recall"],
        "F1_1": report["1"]["f1-score"],
        "Precision_weighted": report["weighted avg"]["precision"],
        "Recall_weighted": report["weighted avg"]["recall"],
        "F1_weighted": report["weighted avg"]["f1-score"],
        "AUC": auc_val,
        "ROC_Curve_Path": str(roc_path),
        "Confusion_Matrix_Path": str(cm_path)
    }
    return metrics_row

results = []

log_reg = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(random_state=42, max_depth=5)

results.append(train_eval(log_reg, "Logistic Regression (No SMOTE)", X_train, y_train, X_test, y_test, "lr_no_smote"))
results.append(train_eval(dt, "Decision Tree (No SMOTE)", X_train, y_train, X_test, y_test, "dt_no_smote"))

# SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

results.append(train_eval(log_reg, "Logistic Regression (SMOTE)", X_res, y_res, X_test, y_test, "lr_smote"))
results.append(train_eval(dt, "Decision Tree (SMOTE)", X_res, y_res, X_test, y_test, "dt_smote"))

# Save metrics
metrics_df = pd.DataFrame(results)
metrics_csv = OUT_DIR / "metrics_comparison_with_auc.csv"
metrics_df.to_csv(metrics_csv, index=False)

from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("Task 4 - Metrics (with AUC)", metrics_df)

print("Saved outputs to:", OUT_DIR)
print("Figures folder:", FIG_DIR)
print("Metrics CSV:", metrics_csv)
print("Unique loan_status values originally:", unique_status)
