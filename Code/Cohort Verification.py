# =============================================================================
# SCRIPT 1: UTEROMAP COHORT VERIFICATION
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION & OUTPUT TREE
# =============================================================================

FILE_PATH = r"uteromap.csv"

OUTPUT_DIR = os.path.dirname(FILE_PATH)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "uteromap_cohort_verification_report.txt"
)


# =============================================================================
# LOGGER
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


sys.stdout = Logger(REPORT_FILE)


# =============================================================================
# 1. LOAD & CLEAN DATA
# =============================================================================

print("=" * 80)
print("SCRIPT 1: UTEROMAP COHORT VERIFICATION")
print("ENGLISH DATASET")
print("=" * 80)


# -----------------------------------------------------------------------------
# Load English CSV
#
# IMPORTANT:
# The original CSV is semicolon-separated, so keep sep=';'
# -----------------------------------------------------------------------------

df = pd.read_csv(
    FILE_PATH,
    sep=';'
)

df['original_row'] = np.arange(1, len(df) + 1)


# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================

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


# =============================================================================
# CHECK REQUIRED COLUMNS
# =============================================================================

print("\nChecking required columns...")

REQUIRED_COLUMNS = [
    TARGET,
    INF_TYPE,
    BIRTHS,
    CSECTION,
    AGE,
    LMP,
    URM
] + MORPH_COLS


missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]


if missing_columns:

    print("\n" + "=" * 80)
    print("ERROR: REQUIRED COLUMNS ARE MISSING")
    print("=" * 80)

    for col in missing_columns:
        print(f"  MISSING: {col}")

    print("\nAvailable columns:")

    for i, col in enumerate(df.columns, start=1):
        print(f"  {i:02d}: {col}")

    raise KeyError(
        "The English CSV does not contain all required columns."
    )


print("All required columns are present.")


# =============================================================================
# BASIC DATASET INFORMATION
# =============================================================================

print("\nDataset dimensions:")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# =============================================================================
# NUMERIC CONVERSIONS
# =============================================================================

df['_target_num'] = pd.to_numeric(
    df[TARGET],
    errors='coerce'
)

df['_births_num'] = pd.to_numeric(
    df[BIRTHS],
    errors='coerce'
)

df['_csection_num'] = pd.to_numeric(
    df[CSECTION],
    errors='coerce'
)

df['_age_num'] = pd.to_numeric(
    df[AGE],
    errors='coerce'
)


# =============================================================================
# INFERTILITY TYPE CLEANING
# =============================================================================

df['_inf_type_stripped'] = (
    df[INF_TYPE]
    .astype('string')
    .str.strip()
)


# =============================================================================
# MORPHOMETRIC NUMERIC CONVERSIONS
# =============================================================================

for col in MORPH_COLS:

    df[f'_{col}_num'] = pd.to_numeric(
        df[col],
        errors='coerce'
    )


# =============================================================================
# CLEAN LMP
# =============================================================================

df['_lmp_num'] = pd.to_numeric(
    df[LMP],
    errors='coerce'
)


# -----------------------------------------------------------------------------
# Reconstruct LMP from the secondary LMP column when necessary
# -----------------------------------------------------------------------------

if URM in df.columns:

    reconstruct_mask = (
            (
                    df[LMP].isna()
                    |
                    df[LMP]
                    .astype(str)
                    .str.strip()
                    .eq('>30 nap')
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

            return (
                    int(nums[0]) * 7
                    +
                    int(nums[1])
            )

        elif len(nums) == 1:

            return int(nums[0]) * 7

        return np.nan


    df.loc[
        reconstruct_mask,
        '_lmp_num'
    ] = (
        df.loc[
            reconstruct_mask,
            URM
        ].apply(extract_urm_days)
    )

# Cap LMP at 40 days
df.loc[
    df['_lmp_num'] > 40,
    '_lmp_num'
] = 40

# =============================================================================
# 2. DEFINE CLINICAL GROUPS & PHASES
# =============================================================================

# -----------------------------------------------------------------------------
# Primary infertility
#
# The original Hungarian script accepted both:
#   Steril I.
#   Seril I.
#
# Keep this exactly the same.
# -----------------------------------------------------------------------------

primary_inf = (
    df['_target_num'].eq(1)
    &
    df['_inf_type_stripped'].isin([
        'Steril I.',
        'Seril I.'
    ])
)


# -----------------------------------------------------------------------------
# Secondary infertility
# -----------------------------------------------------------------------------

secondary_inf = (
    df['_target_num'].eq(1)
    &
    df['_inf_type_stripped']
    .str
    .startswith(
        'Steril II',
        na=False
    )
)


# -----------------------------------------------------------------------------
# Primary control
# -----------------------------------------------------------------------------

primary_ctrl = (
    df['_target_num'].eq(0)
    &
    df['_inf_type_stripped'].isna()
    &
    df['_births_num'].eq(0)
    &
    df['_csection_num'].eq(0)
)


# -----------------------------------------------------------------------------
# Secondary control
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Classified cohort
# -----------------------------------------------------------------------------

classified = (
    primary_inf
    |
    primary_ctrl
    |
    secondary_inf
    |
    secondary_ctrl
)


# -----------------------------------------------------------------------------
# Completeness criteria
# -----------------------------------------------------------------------------

age_complete = df['_age_num'].notna()


morph_complete = (
    df[
        [
            f'_{col}_num'
            for col in MORPH_COLS
        ]
    ]
    .notna()
    .all(axis=1)
)


lmp_complete = df['_lmp_num'].notna()


# =============================================================================
# PHASE DEFINITIONS
# =============================================================================

# -----------------------------------------------------------------------------
# Phase II:
# Classified + complete age + complete morphometry
# -----------------------------------------------------------------------------

phase2_mask = (
    classified
    &
    age_complete
    &
    morph_complete
)


# -----------------------------------------------------------------------------
# Phase I:
# Phase II + complete LMP
# -----------------------------------------------------------------------------

phase1_mask = (
    phase2_mask
    &
    lmp_complete
)


# -----------------------------------------------------------------------------
# Phase III:
# Stratified comparison
# -----------------------------------------------------------------------------

phase3_primary = (
    phase2_mask
    &
    (
        primary_inf
        |
        primary_ctrl
    )
)


phase3_secondary = (
    phase2_mask
    &
    (
        secondary_inf
        |
        secondary_ctrl
    )
)


# =============================================================================
# 3. PRINT COHORT RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("COHORT VERIFICATION RESULTS")
print("=" * 80)


print(
    f"\nTotal Assessed Cohort: "
    f"N = {len(df)}"
)


print(
    f"Classified Study Cohort: "
    f"n = {classified.sum()} "
    f"(Excluded: {len(df) - classified.sum()})"
)


print(
    f"  - Primary Infertility: "
    f"n = {primary_inf.sum()}"
)


print(
    f"  - Primary Control: "
    f"n = {primary_ctrl.sum()}"
)


print(
    f"  - Secondary Infertility: "
    f"n = {secondary_inf.sum()}"
)


print(
    f"  - Secondary Control: "
    f"n = {secondary_ctrl.sum()}"
)


print("-" * 80)


print(
    f"Phase I (Complete Dataset): "
    f"n = {phase1_mask.sum()}"
)


print(
    f"Phase II (Extended Dataset): "
    f"n = {phase2_mask.sum()} "
    f"(Excluded: "
    f"{classified.sum() - phase2_mask.sum()})"
)


print(
    f"Phase III Primary Comparison: "
    f"n = {phase3_primary.sum()} "
    f"(Inf: "
    f"{(phase3_primary & primary_inf).sum()}, "
    f"Ctrl: "
    f"{(phase3_primary & primary_ctrl).sum()})"
)


print(
    f"Phase III Secondary Comparison: "
    f"n = {phase3_secondary.sum()} "
    f"(Inf: "
    f"{(phase3_secondary & secondary_inf).sum()}, "
    f"Ctrl: "
    f"{(phase3_secondary & secondary_ctrl).sum()})"
)


# =============================================================================
# 4. ADDITIONAL DATA QUALITY INFORMATION
# =============================================================================

print("\n" + "=" * 80)
print("DATA COMPLETENESS AUDIT")
print("=" * 80)


print(
    f"\nClassified cohort: "
    f"{classified.sum()}"
)


print(
    f"Classified with complete age: "
    f"{(classified & age_complete).sum()}"
)


print(
    f"Classified with complete morphometry: "
    f"{(classified & morph_complete).sum()}"
)


print(
    f"Classified with complete age + morphometry: "
    f"{phase2_mask.sum()}"
)


print(
    f"Phase II without valid LMP: "
    f"{(phase2_mask & ~lmp_complete).sum()}"
)


print(
    f"Phase I with valid LMP: "
    f"{phase1_mask.sum()}"
)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print(
    f"\nCohort audit log saved to: "
    f"{REPORT_FILE}"
)


print("\n" + "=" * 80)
print("ANALYSIS COMPLETED")
print("=" * 80)


# =============================================================================
# RESTORE STANDARD OUTPUT
# =============================================================================

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
