"""
DEFINITIVE WARFARIN 7-MODEL PIPELINE
=====================================
Uses optimised hyperparameters from manuscript Table 2. The hyperparameter
search that identified these values is implemented in
warfarin_7model_complete.ipynb.

All metrics computed on mg/week scale (back-transformed from sqrt).
Hyperparameter tuning minimised MAE on the square-root scale; all reported
performance metrics were computed after back-transformation.
Negative square-root predictions are clipped to zero before back-transformation.

Produces ALL tables and figures for publication.
Saves everything to Excel workbook + CSVs.

Environment: Python 3.11.9, scikit-learn 1.9.1, pandas 3.0.6,
             NumPy 2.4.6, XGBoost 3.2.0 (see requirements.txt).
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import json

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed, skipping XGBoost model")

SEED = 42
np.random.seed(SEED)

# ── Output directory ──
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN DATA (identical to manuscript)
# ════════════════════════════════════════════════════════════════════════
df_raw = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'warfarin_iwpc.csv'))
if 'Race' in df_raw.columns:
    df_raw.rename(columns={'Race': 'Race (OMB)'}, inplace=True)
df_raw = df_raw[df_raw.iloc[:, 0].notna()]
print(f"Raw dataset: {len(df_raw)} rows")

df = df_raw.copy()
# Inclusion criteria (matching manuscript CONSORT)
n0 = len(df)
df = df[df['Subject Reached Stable Dose of Warfarin'] == 1]
n1 = len(df)
df = df[df['Therapeutic Dose of Warfarin'].notna()]
n2 = len(df)
df = df[df['Therapeutic Dose of Warfarin'] < 300]
n3 = len(df)
df = df[df['CYP2C9 consensus'].notna()]
n4 = len(df)
df = df[df['VKORC1 -1639 consensus'].notna()]
n5 = len(df)
df = df[df['Height (cm)'].notna()]
n6 = len(df)
df = df[df['Weight (kg)'].notna()]
n7 = len(df)
df = df[df['Age'].notna()]
n8 = len(df)

consort = {
    'Total records': n0,
    'After stable dose filter': n1,
    'After dose not null': n2,
    'After dose < 300': n3,
    'After CYP2C9 not null': n4,
    'After VKORC1 not null': n5,
    'After height not null': n6,
    'After weight not null': n7,
    'After age not null': n8,
}

def age_to_ordinal(a):
    try:
        return int(str(a).split('-')[0].strip().split('+')[0]) // 10
    except:
        return np.nan

df['age_ordinal'] = df['Age'].apply(age_to_ordinal)
df = df[df['age_ordinal'].notna()]
n9 = len(df)
consort['After age parsed'] = n9

# Enzyme inducers
for c in ['Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin']:
    df[c] = df[c].fillna(0)
df['enzyme_inducer'] = (df[['Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)',
                             'Rifampin or Rifampicin']].sum(axis=1) > 0).astype(int)
df['Amiodarone (Cordarone)'] = df['Amiodarone (Cordarone)'].fillna(0)

print(f"\nCONSORT Flow:")
for k, v in consort.items():
    print(f"  {k}: {v}")

# ── Split by race ──
race = df['Race (OMB)'].astype(str).str.strip().str.lower()
df_main = df[race != 'asian'].copy()
df_asian = df[race == 'asian'].copy()
print(f"\nMain cohort (non-Asian): {len(df_main)}")
print(f"Asian cohort: {len(df_asian)}")

# ── Features ──
predictors = ['age_ordinal', 'Height (cm)', 'Weight (kg)', 'CYP2C9 consensus',
              'VKORC1 -1639 consensus', 'Race (OMB)', 'enzyme_inducer',
              'Amiodarone (Cordarone)']

def build_X(d):
    return pd.get_dummies(d[predictors],
                          columns=['CYP2C9 consensus', 'VKORC1 -1639 consensus', 'Race (OMB)'])

X_main = build_X(df_main)
y_main_sqrt = np.sqrt(df_main['Therapeutic Dose of Warfarin'].values)
y_main_actual = df_main['Therapeutic Dose of Warfarin'].values

X_asian = build_X(df_asian).reindex(columns=X_main.columns, fill_value=0)
y_asian_actual = df_asian['Therapeutic Dose of Warfarin'].values
y_asian_sqrt = np.sqrt(y_asian_actual)

# ── Train/test split (70/30) ──
X_train, X_test, y_train, y_test_sqrt = train_test_split(
    X_main, y_main_sqrt, test_size=0.3, random_state=SEED
)
y_test_actual = y_test_sqrt ** 2

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
print(f"Asian (all used for external eval): {len(X_asian)}")

# ── Scale for SVR ──
scaler = StandardScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)
X_asian_sc = scaler.transform(X_asian)

# ════════════════════════════════════════════════════════════════════════
# 2. DEFINE MODELS (Table 2 hyperparameters from manuscript)
# ════════════════════════════════════════════════════════════════════════
models = {
    'MLR': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01, max_iter=10000),
    'SVR (RBF)': SVR(kernel='rbf', C=10, gamma=0.01, epsilon=0.2),
    'RF (Default)': RandomForestRegressor(n_estimators=500, random_state=SEED),
    'RF (Tuned)': RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=2,
        min_samples_split=10, max_features='sqrt', random_state=SEED
    ),
}
if HAS_XGB:
    models['XGBoost'] = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=1.0, colsample_bytree=0.8, min_child_weight=5,
        random_state=SEED, verbosity=0
    )

# ════════════════════════════════════════════════════════════════════════
# 3. EVALUATE FUNCTION
# ════════════════════════════════════════════════════════════════════════
def evaluate(y_actual, y_pred_actual):
    """All metrics on mg/week scale."""
    mae = mean_absolute_error(y_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred_actual))
    r2 = r2_score(y_actual, y_pred_actual)
    # P20: fraction within 20% of actual
    pct_diff = np.abs(y_pred_actual - y_actual) / y_actual * 100
    p20 = np.mean(pct_diff <= 20) * 100
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'P20': p20}

# ════════════════════════════════════════════════════════════════════════
# 4. FIT & PREDICT ALL MODELS
# ════════════════════════════════════════════════════════════════════════
results_main = {}
results_asian = {}
predictions_test = {}
predictions_asian = {}

for name, model in models.items():
    use_scaled = 'SVR' in name
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc if use_scaled else X_test
    Xas = X_asian_sc if use_scaled else X_asian

    model.fit(Xtr, y_train)

    # Predict on sqrt scale, back-transform
    pred_sqrt_test = model.predict(Xte)
    pred_sqrt_asian = model.predict(Xas)

    # Clip negative sqrt predictions to 0
    pred_sqrt_test = np.maximum(pred_sqrt_test, 0)
    pred_sqrt_asian = np.maximum(pred_sqrt_asian, 0)

    pred_actual_test = pred_sqrt_test ** 2
    pred_actual_asian = pred_sqrt_asian ** 2

    predictions_test[name] = pred_actual_test
    predictions_asian[name] = pred_actual_asian

    results_main[name] = evaluate(y_test_actual, pred_actual_test)
    results_asian[name] = evaluate(y_asian_actual, pred_actual_asian)

    print(f"\n{name}:")
    print(f"  Main:  MAE={results_main[name]['MAE']:.2f}  RMSE={results_main[name]['RMSE']:.2f}  "
          f"R2={results_main[name]['R2']:.3f}  P20={results_main[name]['P20']:.1f}%")
    print(f"  Asian: MAE={results_asian[name]['MAE']:.2f}  RMSE={results_asian[name]['RMSE']:.2f}  "
          f"R2={results_asian[name]['R2']:.3f}  P20={results_asian[name]['P20']:.1f}%")

# ════════════════════════════════════════════════════════════════════════
# 5. TABLE 3: MAIN PERFORMANCE
# ════════════════════════════════════════════════════════════════════════
table3_rows = []
for name in models:
    m = results_main[name]
    a = results_asian[name]
    table3_rows.append({
        'Model': name,
        'Main MAE': round(m['MAE'], 2),
        'Main RMSE': round(m['RMSE'], 2),
        'Main R2': round(m['R2'], 3),
        'Main P20 (%)': round(m['P20'], 1),
        'Asian MAE': round(a['MAE'], 2),
        'Asian RMSE': round(a['RMSE'], 2),
        'Asian R2': round(a['R2'], 3),
        'Asian P20 (%)': round(a['P20'], 1),
    })
df_table3 = pd.DataFrame(table3_rows)
print("\n\n" + "=" * 100)
print("TABLE 3: MODEL PERFORMANCE")
print("=" * 100)
print(df_table3.to_string(index=False))

# ════════════════════════════════════════════════════════════════════════
# 6. BOOTSTRAP 95% CIs (1000 iterations)
# ════════════════════════════════════════════════════════════════════════
print("\n\nComputing bootstrap CIs (2000 iterations)...")
N_BOOT = 2000
boot_results_main = {name: {'MAE': [], 'RMSE': [], 'R2': [], 'P20': []} for name in models}
boot_results_asian = {name: {'MAE': [], 'RMSE': [], 'R2': [], 'P20': []} for name in models}
boot_delta_mae = {name: [] for name in models}  # ΔMAE vs MLR

n_test = len(y_test_actual)
n_asian = len(y_asian_actual)

for b in range(N_BOOT):
    idx_test = np.random.choice(n_test, n_test, replace=True)
    idx_asian = np.random.choice(n_asian, n_asian, replace=True)

    for name in models:
        # Main
        m = evaluate(y_test_actual[idx_test], predictions_test[name][idx_test])
        for k in ['MAE', 'RMSE', 'R2', 'P20']:
            boot_results_main[name][k].append(m[k])

        # Asian
        a = evaluate(y_asian_actual[idx_asian], predictions_asian[name][idx_asian])
        for k in ['MAE', 'RMSE', 'R2', 'P20']:
            boot_results_asian[name][k].append(a[k])

        # ΔMAE vs MLR (main)
        mlr_mae = mean_absolute_error(y_test_actual[idx_test], predictions_test['MLR'][idx_test])
        boot_delta_mae[name].append(m['MAE'] - mlr_mae)

# Build CI table
ci_rows = []
for name in models:
    row = {'Model': name}
    for metric in ['MAE', 'RMSE', 'R2', 'P20']:
        vals = boot_results_main[name][metric]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        row[f'Main {metric} [95% CI]'] = f"{np.mean(vals):.2f} [{lo:.2f}, {hi:.2f}]"
    for metric in ['MAE', 'RMSE', 'R2', 'P20']:
        vals = boot_results_asian[name][metric]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        row[f'Asian {metric} [95% CI]'] = f"{np.mean(vals):.2f} [{lo:.2f}, {hi:.2f}]"
    ci_rows.append(row)
df_boot = pd.DataFrame(ci_rows)

# ════════════════════════════════════════════════════════════════════════
# 7. TABLE S2: PAIRED BOOTSTRAP ΔMAE vs MLR
# ════════════════════════════════════════════════════════════════════════
paired_rows = []
for name in models:
    if name == 'MLR':
        continue
    deltas = boot_delta_mae[name]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    sig = "Yes" if (lo > 0 or hi < 0) else "No"
    paired_rows.append({
        'Model vs MLR': name,
        'dMAE': f"{np.mean(deltas):+.3f}",
        '95% CI': f"[{lo:+.3f}, {hi:+.3f}]",
        'Significant': sig
    })
df_paired = pd.DataFrame(paired_rows)
print("\n\nTABLE S2: PAIRED BOOTSTRAP dMAE vs MLR")
print(df_paired.to_string(index=False))

# ════════════════════════════════════════════════════════════════════════
# 8. TABLE S1: 5-FOLD CROSS-VALIDATION
# ════════════════════════════════════════════════════════════════════════
print("\n\nComputing 5-fold CV...")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rows = []

for name, model_template in models.items():
    # We need fresh model instances
    import copy
    model_cv = copy.deepcopy(model_template)

    use_scaled = 'SVR' in name
    X_all = X_main.values
    y_all = y_main_sqrt

    maes, rmses, r2s, p20s = [], [], [], []
    for train_idx, val_idx in kf.split(X_all):
        Xtr_cv, Xval_cv = X_all[train_idx], X_all[val_idx]
        ytr_cv, yval_cv_sqrt = y_all[train_idx], y_all[val_idx]
        yval_actual = yval_cv_sqrt ** 2

        # Scale inside fold for SVR to prevent information leakage
        if use_scaled:
            fold_scaler = StandardScaler().fit(Xtr_cv)
            Xtr_cv = fold_scaler.transform(Xtr_cv)
            Xval_cv = fold_scaler.transform(Xval_cv)

        model_fresh = copy.deepcopy(model_template)
        model_fresh.fit(Xtr_cv, ytr_cv)
        pred_sqrt = np.maximum(model_fresh.predict(Xval_cv), 0)
        pred_actual = pred_sqrt ** 2

        ev = evaluate(yval_actual, pred_actual)
        maes.append(ev['MAE'])
        rmses.append(ev['RMSE'])
        r2s.append(ev['R2'])
        p20s.append(ev['P20'])

    cv_rows.append({
        'Model': name,
        'CV MAE': f"{np.mean(maes):.2f} ± {np.std(maes):.2f}",
        'CV RMSE': f"{np.mean(rmses):.2f} ± {np.std(rmses):.2f}",
        'CV R2': f"{np.mean(r2s):.3f} ± {np.std(r2s):.3f}",
        'CV P20 (%)': f"{np.mean(p20s):.1f} ± {np.std(p20s):.1f}",
    })
df_cv = pd.DataFrame(cv_rows)
print(df_cv.to_string(index=False))

# ════════════════════════════════════════════════════════════════════════
# 9. TABLE S3: CALIBRATION (slope, intercept)
# ════════════════════════════════════════════════════════════════════════
cal_rows = []
for name in models:
    for cohort, y_act, y_pred in [('Main', y_test_actual, predictions_test[name]),
                                    ('Asian', y_asian_actual, predictions_asian[name])]:
        from numpy.polynomial.polynomial import polyfit
        # Calibration: actual = intercept + slope * predicted
        slope, intercept = np.polyfit(y_pred, y_act, 1)
        cal_rows.append({
            'Model': name,
            'Cohort': cohort,
            'Calibration Slope': round(slope, 3),
            'Calibration Intercept': round(intercept, 2),
        })
df_cal = pd.DataFrame(cal_rows)

# ════════════════════════════════════════════════════════════════════════
# 10. TABLE S4: DOSE-STRATIFIED PERFORMANCE
# ════════════════════════════════════════════════════════════════════════
dose_bins = [(0, 21, 'Low (<21)'), (21, 49, 'Medium (21-49)'), (49, 300, 'High (>=49)')]
strat_rows = []
for name in models:
    for lo, hi, label in dose_bins:
        mask = (y_test_actual >= lo) & (y_test_actual < hi)
        if mask.sum() < 5:
            continue
        ev = evaluate(y_test_actual[mask], predictions_test[name][mask])
        strat_rows.append({
            'Model': name,
            'Dose Range': label,
            'N': int(mask.sum()),
            'MAE': round(ev['MAE'], 2),
            'RMSE': round(ev['RMSE'], 2),
            'R2': round(ev['R2'], 3),
            'P20 (%)': round(ev['P20'], 1),
        })
df_strat = pd.DataFrame(strat_rows)

# ════════════════════════════════════════════════════════════════════════
# 11. TABLE S5: MISSINGNESS AUDIT
# ════════════════════════════════════════════════════════════════════════
miss_cols = ['Height (cm)', 'Weight (kg)', 'Age', 'CYP2C9 consensus',
             'VKORC1 -1639 consensus', 'Race (OMB)', 'Amiodarone (Cordarone)',
             'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)',
             'Rifampin or Rifampicin', 'Therapeutic Dose of Warfarin',
             'Subject Reached Stable Dose of Warfarin']
miss_rows = []
for c in miss_cols:
    if c in df_raw.columns:
        total = len(df_raw)
        missing = df_raw[c].isna().sum()
        miss_rows.append({
            'Variable': c,
            'Total N': total,
            'Missing N': missing,
            'Missing %': round(missing / total * 100, 1),
        })
df_miss = pd.DataFrame(miss_rows)

# ════════════════════════════════════════════════════════════════════════
# 12. TABLE S6: MLR COEFFICIENTS
# ════════════════════════════════════════════════════════════════════════
mlr_model = models['MLR']
coef_rows = []
for feat, coef in zip(X_train.columns, mlr_model.coef_):
    coef_rows.append({'Feature': feat, 'Coefficient': round(coef, 4)})
coef_rows.append({'Feature': '(Intercept)', 'Coefficient': round(mlr_model.intercept_, 4)})
df_coefs = pd.DataFrame(coef_rows)

# ════════════════════════════════════════════════════════════════════════
# 13. FIGURE DATA: SCATTER PLOTS
# ════════════════════════════════════════════════════════════════════════
scatter_main = pd.DataFrame({
    'Actual': y_test_actual,
    **{name: predictions_test[name] for name in models}
})
scatter_asian = pd.DataFrame({
    'Actual': y_asian_actual,
    **{name: predictions_asian[name] for name in models}
})

# ════════════════════════════════════════════════════════════════════════
# 14. FIGURE DATA: RF PERMUTATION IMPORTANCE
# ════════════════════════════════════════════════════════════════════════
rf_tuned = models['RF (Tuned)']
imp_rows = []
from sklearn.inspection import permutation_importance
perm_result = permutation_importance(rf_tuned, X_test.values, y_test_sqrt,
                                     n_repeats=30, random_state=SEED,
                                     scoring='neg_mean_absolute_error')
for feat, imp in zip(X_train.columns, perm_result.importances_mean):
    imp_rows.append({'Feature': feat, 'Importance': round(abs(imp), 4)})
df_imp = pd.DataFrame(imp_rows).sort_values('Importance', ascending=False)

# ════════════════════════════════════════════════════════════════════════
# 15. SAVE EVERYTHING TO EXCEL + CSVs
# ════════════════════════════════════════════════════════════════════════
print("\n\nSaving all results...")

# Individual CSVs
df_table3.to_csv(os.path.join(OUT_DIR, 'Table3_Performance.csv'), index=False)
df_boot.to_csv(os.path.join(OUT_DIR, 'Table3B_Bootstrap_CIs.csv'), index=False)
df_cv.to_csv(os.path.join(OUT_DIR, 'Table_S1_CrossValidation.csv'), index=False)
df_paired.to_csv(os.path.join(OUT_DIR, 'Table_S2_Paired_Bootstrap.csv'), index=False)
df_cal.to_csv(os.path.join(OUT_DIR, 'Table_S3_Calibration.csv'), index=False)
df_strat.to_csv(os.path.join(OUT_DIR, 'Table_S4_Dose_Stratified.csv'), index=False)
df_miss.to_csv(os.path.join(OUT_DIR, 'Table_S5_Missingness.csv'), index=False)
df_coefs.to_csv(os.path.join(OUT_DIR, 'Table_S6_MLR_Coefficients.csv'), index=False)
scatter_main.to_csv(os.path.join(OUT_DIR, 'Fig5_Main_Scatter.csv'), index=False)
scatter_asian.to_csv(os.path.join(OUT_DIR, 'Fig5_Asian_Scatter.csv'), index=False)
df_coefs.to_csv(os.path.join(OUT_DIR, 'Fig6A_MLR_Coefs.csv'), index=False)
df_imp.to_csv(os.path.join(OUT_DIR, 'Fig6B_RF_Importance.csv'), index=False)

# CONSORT
consort_df = pd.DataFrame(list(consort.items()), columns=['Step', 'N'])
consort_df.to_csv(os.path.join(OUT_DIR, 'CONSORT_Flow.csv'), index=False)

# MASTER EXCEL WORKBOOK
xlsx_path = os.path.join(OUT_DIR, 'Warfarin_ALL_Results.xlsx')
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    df_table3.to_excel(writer, sheet_name='Table3_Performance', index=False)
    df_boot.to_excel(writer, sheet_name='Table3B_Bootstrap_CIs', index=False)
    df_cv.to_excel(writer, sheet_name='TableS1_CrossValidation', index=False)
    df_paired.to_excel(writer, sheet_name='TableS2_PairedBootstrap', index=False)
    df_cal.to_excel(writer, sheet_name='TableS3_Calibration', index=False)
    df_strat.to_excel(writer, sheet_name='TableS4_DoseStratified', index=False)
    df_miss.to_excel(writer, sheet_name='TableS5_Missingness', index=False)
    df_coefs.to_excel(writer, sheet_name='TableS6_MLR_Coefficients', index=False)
    df_imp.to_excel(writer, sheet_name='Fig6B_RF_Importance', index=False)
    scatter_main.to_excel(writer, sheet_name='Fig5_MainScatter', index=False)
    scatter_asian.to_excel(writer, sheet_name='Fig5_AsianScatter', index=False)
    consort_df.to_excel(writer, sheet_name='CONSORT_Flow', index=False)

print(f"\n✓ Master Excel: {xlsx_path}")
print(f"✓ All CSVs saved to: {OUT_DIR}")

# ════════════════════════════════════════════════════════════════════════
# 16. SUMMARY FOR MANUSCRIPT
# ════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("FINAL RESULTS FOR MANUSCRIPT (all metrics on mg/week scale)")
print("=" * 100)
print(df_table3.to_string(index=False))
print("\n\nBOOTSTRAP 95% CIs:")
print(df_boot.to_string(index=False))
print("\n\nPAIRED BOOTSTRAP vs MLR:")
print(df_paired.to_string(index=False))
print("\n\n5-FOLD CV:")
print(df_cv.to_string(index=False))
print("\n\nCALIBRATION:")
print(df_cal.to_string(index=False))
print("\n\nDOSE-STRATIFIED:")
print(df_strat.to_string(index=False))
print("\n\nMLR COEFFICIENTS:")
print(df_coefs.to_string(index=False))
print("\n\nRF IMPORTANCE (top 10):")
print(df_imp.head(10).to_string(index=False))

print(f"\n\nDONE. All results saved to {OUT_DIR}")
print(f"Excel workbook: {xlsx_path}")
