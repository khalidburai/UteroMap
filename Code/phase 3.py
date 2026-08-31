# =============================================================================
# SCRIPT 4: PHASE III STRATIFIED SUBGROUP MACHINE LEARNING PIPELINE
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# =============================================================================
# CONFIGURATION & OUTPUT TREE
# =============================================================================

FILE_PATH = r"uteromap.csv"

OUTPUT_DIR = os.path.dirname(FILE_PATH)

LOG_FILE = os.path.join(
    OUTPUT_DIR,
    "Phase3_ML_Console_Report.txt"
)


# =============================================================================
# LOGGER SETUP
# =============================================================================

class Logger(object):

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(LOG_FILE)

print("=" * 90)
print("PHASE III: STRATIFIED SUBGROUP MACHINE LEARNING PIPELINE (ENGLISH)")
print("=" * 90)


# =============================================================================
# 1. LOAD & SPLIT INTO STRATIFIED SUBGROUPS
# =============================================================================

df = pd.read_csv(
    FILE_PATH,
    sep=';'
)

TARGET = 'infertile? Yes = 1 No = 0'
INF_TYPE = 'infertile? Primary infertility / Secondary infertility'
BIRTHS = 'births'
CSECTION = 'C-section'
AGE = 'age'

MORPH_COLS = [
    'u1',
    'u2',
    'u3 co',
    'u3 cx',
    'u3d',
    'wa',
    'e',
    'wp',
    'u3/u2'
]

df['_target_num'] = pd.to_numeric(df[TARGET], errors='coerce')
df['_births_num'] = pd.to_numeric(df[BIRTHS], errors='coerce')
df['_csection_num'] = pd.to_numeric(df[CSECTION], errors='coerce')
df['_age_num'] = pd.to_numeric(df[AGE], errors='coerce')
df['_inf_type_stripped'] = df[INF_TYPE].astype('string').str.strip()

for col in MORPH_COLS:
    df[f'_{col}_num'] = pd.to_numeric(df[col], errors='coerce')

primary_inf = (
    df['_target_num'].eq(1)
    &
    df['_inf_type_stripped'].isin([
        'Steril I.',
        'Seril I.'
    ])
)

secondary_inf = (
    df['_target_num'].eq(1)
    &
    df['_inf_type_stripped'].str.startswith(
        'Steril II',
        na=False
    )
)

primary_ctrl = (
    df['_target_num'].eq(0)
    &
    df['_inf_type_stripped'].isna()
    &
    df['_births_num'].eq(0)
    &
    df['_csection_num'].eq(0)
)

secondary_ctrl = (
    df['_target_num'].eq(0)
    &
    df['_inf_type_stripped'].isna()
    &
    (
        df['_births_num'].gt(0)
        |
        df['_csection_num'].gt(0)
    )
)

classified = (
    primary_inf
    |
    primary_ctrl
    |
    secondary_inf
    |
    secondary_ctrl
)

age_complete = df['_age_num'].notna()

morph_complete = (
    df[[f'_{col}_num' for col in MORPH_COLS]]
    .notna()
    .all(axis=1)
)

phase2_mask = classified & age_complete & morph_complete

df_primary = df[phase2_mask & (primary_inf | primary_ctrl)].copy()
df_secondary = df[phase2_mask & (secondary_inf | secondary_ctrl)].copy()

feature_cols = ['_age_num'] + [f'_{c}_num' for c in MORPH_COLS]

print(
    f"\n1. Primary Comparison Subgroup:   n = {len(df_primary)} "
    f"(Inf: {(df_primary['_target_num'] == 1).sum()}, "
    f"Ctrl: {(df_primary['_target_num'] == 0).sum()})"
)

print(
    f"2. Secondary Comparison Subgroup: n = {len(df_secondary)} "
    f"(Inf: {(df_secondary['_target_num'] == 1).sum()}, "
    f"Ctrl: {(df_secondary['_target_num'] == 0).sum()})"
)


# =============================================================================
# 2. EVALUATION FUNCTION FOR SUBGROUPS
# =============================================================================

def evaluate_subgroup(sub_df, subgroup_name):
    print("\n" + "-" * 85)
    print(f"EVALUATION: {subgroup_name.upper()}")
    print("-" * 85)

    X_sub = sub_df[feature_cols]
    y_sub = sub_df['_target_num'].astype(int)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.01,
            random_state=42
        )
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    sub_results = []
    sub_cms = {}
    sub_rocs = {}

    for name, clf in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('classifier', clf)
        ])

        y_true_all, y_pred_all, y_proba_all = [], [], []

        for train_idx, test_idx in cv.split(X_sub, y_sub):
            X_train, X_test = X_sub.iloc[train_idx], X_sub.iloc[test_idx]
            y_train, y_test = y_sub.iloc[train_idx], y_sub.iloc[test_idx]

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = (
                pipe.predict_proba(X_test)[:, 1]
                if hasattr(pipe, 'predict_proba')
                else y_pred
            )

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_proba_all.extend(y_proba)

        cm = confusion_matrix(y_true_all, y_pred_all)
        sub_cms[name] = cm
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_true_all, y_pred_all)
        bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
        sens = recall_score(y_true_all, y_pred_all)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = precision_score(y_true_all, y_pred_all) if (tp + fp) > 0 else 0
        f1 = f1_score(y_true_all, y_pred_all)
        auc_val = roc_auc_score(y_true_all, y_proba_all)

        fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
        sub_rocs[name] = (fpr, tpr, auc_val)

        sub_results.append({
            'Model': name,
            'Accuracy': acc,
            'Balanced Acc': bal_acc,
            'Sensitivity (Recall)': sens,
            'Specificity': spec,
            'Precision (PPV)': prec,
            'F1 Score': f1,
            'ROC-AUC': auc_val
        })

    df_res = pd.DataFrame(sub_results)
    print(df_res.to_string(index=False))
    return df_res, sub_cms, sub_rocs


prim_df, prim_cms, prim_rocs = evaluate_subgroup(
    df_primary,
    f"Primary Infertility vs. Nulligravid Control (n = {len(df_primary)})"
)

sec_df, sec_cms, sec_rocs = evaluate_subgroup(
    df_secondary,
    f"Secondary Infertility vs. Parous Control (n = {len(df_secondary)})"
)


# =============================================================================
# 3. GENERATE & SAVE PHASE III FIGURES (ENGLISH)
# =============================================================================

# 1. Paired ROC Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for name, (fpr, tpr, auc_val) in prim_rocs.items():
    ax1.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (0.500)')
ax1.set_title(f'Primary Infertility vs. Control (n = {len(df_primary)})', fontweight='bold')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(loc='lower right')

for name, (fpr, tpr, auc_val) in sec_rocs.items():
    ax2.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (0.500)')
ax2.set_title(f'Secondary Infertility vs. Control (n = {len(df_secondary)})', fontweight='bold')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend(loc='lower right')

plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "phase3_roc_comparison_English.png")
plt.savefig(roc_path, dpi=300)
plt.close()

# 2. Confusion Matrices Comparison
fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))

sns.heatmap(
    prim_cms['Random Forest'],
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[0, 0],
    cbar=False,
    xticklabels=['Control (0)', 'Infertile (1)'],
    yticklabels=['Control (0)', 'Infertile (1)']
)
axes[0, 0].set_title(f'Primary: Random Forest (n = {len(df_primary)})', fontweight='bold')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('True')

sns.heatmap(
    prim_cms['Gradient Boosting'],
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[0, 1],
    cbar=False,
    xticklabels=['Control (0)', 'Infertile (1)'],
    yticklabels=['Control (0)', 'Infertile (1)']
)
axes[0, 1].set_title(f'Primary: Gradient Boosting (n = {len(df_primary)})', fontweight='bold')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('True')

sns.heatmap(
    sec_cms['Logistic Regression'],
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[1, 0],
    cbar=False,
    xticklabels=['Control (0)', 'Infertile (1)'],
    yticklabels=['Control (0)', 'Infertile (1)']
)
axes[1, 0].set_title(f'Secondary: Logistic Regression (n = {len(df_secondary)})', fontweight='bold')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('True')

sns.heatmap(
    sec_cms['Naive Bayes'],
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[1, 1],
    cbar=False,
    xticklabels=['Control (0)', 'Infertile (1)'],
    yticklabels=['Control (0)', 'Infertile (1)']
)
axes[1, 1].set_title(f'Secondary: Naive Bayes (n = {len(df_secondary)})', fontweight='bold')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('True')

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "phase3_confusion_matrices_English.png")
plt.savefig(cm_path, dpi=300)
plt.close()

print(f"\nOutputs saved:")
print(f" - Report: {LOG_FILE}")
print(f" - Figures: {roc_path}, {cm_path}")

# =============================================================================
# RESTORE STANDARD OUTPUT
# =============================================================================

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
