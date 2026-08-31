# =============================================================================
# SCRIPT 2: PHASE I MACHINE LEARNING PIPELINE
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

FILE_PATH = r"D:\Ph.D\First-Year\1st-Semester\Project\Infertility\Infertility_Analysis\NewCalculations\uteromap_EnglishCORRECT.csv"

OUTPUT_DIR = os.path.dirname(FILE_PATH)

LOG_FILE = os.path.join(
    OUTPUT_DIR,
    "Phase1_ML_Console_Report_English.txt"
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
print("PHASE I: MACHINE LEARNING PIPELINE (COMPLETE DATASET WITH LMP, ENGLISH)")
print("=" * 90)


# =============================================================================
# 1. LOAD & PREPROCESS DATA USING VERIFIED COHORT LOGIC
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
LMP = 'day of last menstrual period'
URM = 'last menstrual period (LMP)'

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

df['_lmp_num'] = pd.to_numeric(df[LMP], errors='coerce')

if URM in df.columns:
    reconstruct_mask = (
        (
            df[LMP].isna()
            |
            df[LMP].astype(str).str.strip().eq('>30 nap')
        )
        &
        df[URM].notna()
    )

    def extract_urm_days(value):
        if pd.isna(value):
            return np.nan

        nums = (
            pd.Series(str(value))
            .str.findall(r'\d+')
            .iloc[0]
        )

        if len(nums) >= 2:
            return int(nums[0]) * 7 + int(nums[1])
        elif len(nums) == 1:
            return int(nums[0]) * 7
        return np.nan

    df.loc[reconstruct_mask, '_lmp_num'] = (
        df.loc[reconstruct_mask, URM].apply(extract_urm_days)
    )

df.loc[df['_lmp_num'] > 40, '_lmp_num'] = 40

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

lmp_complete = df['_lmp_num'].notna()

phase1_df = df[
    classified
    &
    age_complete
    &
    morph_complete
    &
    lmp_complete
].copy()

feature_names = [
    'age',
    'day of last menstrual period',
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

feature_cols = ['_age_num', '_lmp_num'] + [f'_{c}_num' for c in MORPH_COLS]

X = phase1_df[feature_cols]
y = phase1_df['_target_num'].astype(int)

print(
    f"\nCohort Size: n = {len(X)} | "
    f"Fertile Controls (0): {(y == 0).sum()} | "
    f"Infertile Cases (1): {(y == 1).sum()}"
)


# =============================================================================
# 2. LEAK-FREE CLASSIFICATION WITH STRATIFIED 10-FOLD CV
# =============================================================================

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

results = []
conf_matrices = {}
roc_data = {}

for name, clf in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('classifier', clf)
    ])

    y_true_all, y_pred_all, y_proba_all = [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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
    conf_matrices[name] = cm
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true_all, y_pred_all)
    bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
    sens = recall_score(y_true_all, y_pred_all)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = precision_score(y_true_all, y_pred_all) if (tp + fp) > 0 else 0
    f1 = f1_score(y_true_all, y_pred_all)
    auc_val = roc_auc_score(y_true_all, y_proba_all)

    fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
    roc_data[name] = (fpr, tpr, auc_val)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Balanced Acc': bal_acc,
        'Sensitivity (Recall)': sens,
        'Specificity': spec,
        'Precision (PPV)': prec,
        'F1 Score': f1,
        'ROC-AUC': auc_val
    })

res_df = pd.DataFrame(results)

print("\n" + "=" * 90)
print("PHASE I PERFORMANCE METRICS TABLE (ENGLISH)")
print("=" * 90)
print(res_df.to_string(index=False))


# =============================================================================
# 3. GENERATE & SAVE PLOTS
# =============================================================================

# 1. ROC Curves
plt.figure(figsize=(7.5, 6))
for name, (fpr, tpr, auc_val) in roc_data.items():
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (AUC = 0.500)')
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
plt.title('Phase I: ROC Curves Comparison (English Dataset)', fontsize=12, fontweight='bold')
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()

roc_path = os.path.join(OUTPUT_DIR, "phase1_roc_curves_English.png")
plt.savefig(roc_path, dpi=300)
plt.close()

# 2. Confusion Matrices (Random Forest & Gradient Boosting)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for i, m_name in enumerate(['Random Forest', 'Gradient Boosting']):
    sns.heatmap(
        conf_matrices[m_name],
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[i],
        cbar=False,
        xticklabels=['Control (0)', 'Infertile (1)'],
        yticklabels=['Control (0)', 'Infertile (1)']
    )
    axes[i].set_title(f'Phase I: {m_name}', fontweight='bold')
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "phase1_confusion_matrices_English.png")
plt.savefig(cm_path, dpi=300)
plt.close()

# 3. Feature Importance
rf_raw = RandomForestClassifier(n_estimators=100, random_state=42)
rf_raw.fit(X, y)

feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Gini Importance': rf_raw.feature_importances_
}).sort_values(by='Gini Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_imp, x='Gini Importance', y='Feature', palette='viridis')
plt.title('Phase I: Random Forest Feature Importances', fontsize=12, fontweight='bold')
plt.xlabel('Gini Importance')
plt.ylabel('Feature')
plt.tight_layout()

fi_path = os.path.join(OUTPUT_DIR, "phase1_feature_importances_English.png")
plt.savefig(fi_path, dpi=300)
plt.close()

print(f"\nOutputs saved:")
print(f" - Report: {LOG_FILE}")
print(f" - Figures: {roc_path}, {cm_path}, {fi_path}")

# =============================================================================
# RESTORE STANDARD OUTPUT
# =============================================================================

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal