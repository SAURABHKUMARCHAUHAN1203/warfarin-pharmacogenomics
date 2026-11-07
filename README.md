# Warfarin Pharmacogenomics: Comparative Machine Learning Analysis

**Comparative evaluation of machine learning algorithms for warfarin dose prediction using the IWPC pharmacogenomics dataset.**

## Authors

- Amir Sams¹ (Joint First Author)
- Saurabh Kumar Chauhan¹* (Joint First / Co-Corresponding) — ORCID: 0009-0004-8496-8445
- Maliha Khan¹
- Suruchi Singh¹* (Senior / Co-Corresponding) — ORCID: 0000-0001-8553-6725

¹ Department of Pharmaceutical Sciences, Sam Higginbottom University of Agriculture, Technology and Sciences (SHUATS), Prayagraj, India

## Overview

This repository contains the complete analysis pipeline for a 7-model comparison of warfarin dose prediction algorithms:

1. **Multiple Linear Regression (MLR)**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Support Vector Regression (SVR, RBF kernel)**
5. **Random Forest (default hyperparameters)**
6. **Random Forest (tuned hyperparameters)**
7. **XGBoost**

Models are trained on sqrt-transformed weekly warfarin dose (mg/week), with performance metrics (MAE, RMSE, R², P20) computed on the back-transformed mg/week scale.

## Repository Structure

```
├── data/
│   └── warfarin_iwpc.csv              # IWPC warfarin dataset (5700 patients)
├── output/
│   ├── Warfarin_ALL_Results.xlsx      # Master Excel workbook (all tables)
│   ├── Table3_Performance.csv         # Main performance metrics (7 models)
│   ├── Table3B_Bootstrap_CIs.csv      # Bootstrap 95% CIs (1000 iterations)
│   ├── Table_S1_CrossValidation.csv   # 5-fold cross-validation
│   ├── Table_S2_Paired_Bootstrap.csv  # Paired bootstrap ΔMAE vs MLR
│   ├── Table_S3_Calibration.csv       # Calibration slope/intercept
│   ├── Table_S4_Dose_Stratified.csv   # Low/Medium/High dose performance
│   ├── Table_S5_Missingness.csv       # Missing data audit
│   ├── Table_S6_MLR_Coefficients.csv  # Full MLR coefficients
│   ├── Fig5_Main_Scatter.csv          # Scatter plot data (Main cohort)
│   ├── Fig5_Asian_Scatter.csv         # Scatter plot data (Asian cohort)
│   ├── Fig6A_MLR_Coefs.csv            # MLR coefficient bar chart data
│   ├── Fig6B_RF_Importance.csv        # RF feature importance data
│   └── CONSORT_Flow.csv               # Patient flow numbers
├── warfarin_mega_pipeline.py          # Complete 7-model pipeline script
├── warfarin_analysis_final.ipynb      # Original 2-model notebook (Colab)
└── README.md
```

## Quick Start

```bash
pip install pandas numpy scikit-learn xgboost openpyxl
python warfarin_mega_pipeline.py
```

All output CSVs and the master Excel workbook will be saved to `output/`.

## Key Results

| Model | Main MAE (mg/wk) | Main R² | Asian MAE (mg/wk) | Asian R² |
|---|---|---|---|---|
| MLR | 9.43 | 0.391 | 6.70 | 0.151 |
| Ridge | 9.43 | 0.389 | 6.69 | 0.153 |
| Lasso | 9.58 | 0.343 | 6.64 | 0.164 |
| SVR (RBF) | 9.51 | 0.369 | 6.47 | 0.206 |
| RF (Default) | 10.23 | 0.295 | 7.21 | 0.047 |
| RF (Tuned) | 9.75 | 0.339 | 6.70 | 0.165 |
| XGBoost | 9.67 | 0.357 | 6.64 | 0.170 |

**Conclusion:** No ML model significantly outperforms multiple linear regression on the ethnically diverse main cohort. SVR (RBF) shows the best Asian-subgroup performance (MAE 6.47 mg/wk, R² 0.206).

## Dataset

The IWPC (International Warfarin Pharmacogenetics Consortium) dataset contains 5,700 patients from 21 research groups across 9 countries. After quality filters (stable dose, complete genotype/phenotype data), 3,449 patients remain: 2,329 non-Asian (main cohort) and 1,120 Asian.

## License

This project is for academic/research purposes.

## Citation

If you use this code, please cite:
> Sams A, Chauhan SK, Khan M, Singh S. Comparative evaluation of machine learning algorithms for warfarin dose prediction using pharmacogenomics data. *Pharmacol Res Perspect*. 2026.
