import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import patsy

# =========================
# Load data
# =========================
df = pd.read_excel("neurons4GLM.xlsx")

features = [
    'onP_amp', 'on_latency', 'on_charge', 'offP_amp_pla', 'offP_amp_bas',
    'off_latency', 'off_charge', 'sustain', 'inhibit_early',
    'inhibit_late', 'inhibit_off'
]

# =========================
# Z-score features per neuron (filename)
# =========================
df_z = df.copy()
df_z[features] = df_z.groupby('filename')[features].transform(lambda x: (x - x.mean()) / x.std(ddof=0))

# =========================
# Predictors
# =========================
groups = df_z['mouseID'].values
logo = LeaveOneGroupOut()

# =========================
# Helper function for LONO R²
# =========================
def lono_r2_sklearn(X, y, groups, model=None):
    if model is None:
        model = LinearRegression()
    test_r2s = []
    for train_idx, test_idx in logo.split(X, y, groups):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        test_r2s.append(r2_score(yte, ypred))
    return np.mean(test_r2s), np.array(test_r2s)

# =========================
# Interaction terms
# =========================
interaction_terms = ['para1:x','para1:y','para2:x','para2:y','para3:x','para3:y']

# =========================
# 1. Linear Regression with interactions
# =========================
linear_results = []

for feat in features:
    y = df_z[feat].values
    X_base = df_z[['para1','para2','para3','x','y']].values
    X_full = np.hstack([
        X_base,
        (df_z['para1']*df_z['x']).values[:,None],
        (df_z['para1']*df_z['y']).values[:,None],
        (df_z['para2']*df_z['x']).values[:,None],
        (df_z['para2']*df_z['y']).values[:,None],
        (df_z['para3']*df_z['x']).values[:,None],
        (df_z['para3']*df_z['y']).values[:,None],
    ])

    r2_base, _ = lono_r2_sklearn(X_base, y, groups)
    r2_full, _ = lono_r2_sklearn(X_full, y, groups)

    linear_results.append({
        'feature': feat,
        'Base_LONO_R2': r2_base,
        'Full_with_interactions_LONO_R2': r2_full,
        'Delta_R2': r2_full - r2_base
    })

linear_df = pd.DataFrame(linear_results)
linear_df.to_excel("lsfm_GLM_LONO_linear_interactions_summary.xlsx", index=False)

# =========================
# 2. Mixed Effects with interactions
# =========================
mixed_results = []
mixed_pvals = []

for feat in features:
    formula = (f"{feat} ~ para1 + para2 + para3 + x + y "
               "+ para1:x + para1:y + para2:x + para2:y + para3:x + para3:y")

    # LONO R²
    test_r2s = []
    for train_idx, test_idx in logo.split(df_z, df_z[feat], groups):
        df_tr = df_z.iloc[train_idx]
        df_te = df_z.iloc[test_idx]

        md = MixedLM.from_formula(formula, groups='mouseID', data=df_tr)
        mdf = md.fit(reml=False, method="lbfgs")

        fe_params = mdf.fe_params
        Xte_design = patsy.dmatrix(formula.split("~",1)[1], data=df_te, return_type='dataframe')
        Xte_design = Xte_design[fe_params.index]
        ypred = np.dot(Xte_design, fe_params.values)

        yte = df_te[feat].values
        test_r2s.append(r2_score(yte, ypred))

    mixed_results.append({
        'feature': feat,
        'Mixed_LONO_R2_with_interactions': np.mean(test_r2s)
    })

    # p-values from full dataset fit
    md_full = MixedLM.from_formula(formula, groups='mouseID', data=df_z)
    mdf_full = md_full.fit(reml=False, method="lbfgs")
    pvals = {term: mdf_full.pvalues.get(term, np.nan) for term in interaction_terms}
    row = {'feature': feat}
    row.update(pvals)
    mixed_pvals.append(row)

mixed_df = pd.DataFrame(mixed_results)
mixed_df.to_excel("lsfm_GLM_LONO_mixed_interactions_summary.xlsx", index=False)

pval_df = pd.DataFrame(mixed_pvals)
pval_df.to_excel("para_location_interactions_pvals.xlsx", index=False)

# =========================
# 3. GLM with interactions
# =========================
glm_results = []
glm_pvals = []

for feat in features:
    formula = (f"{feat} ~ para1 + para2 + para3 + x + y "
               "+ para1:x + para1:y + para2:x + para2:y + para3:x + para3:y")

    # LONO R²
    test_r2s = []
    for train_idx, test_idx in logo.split(df_z, df_z[feat], groups):
        df_tr = df_z.iloc[train_idx]
        df_te = df_z.iloc[test_idx]

        model = smf.glm(formula=formula, data=df_tr, family=sm.families.Gaussian())
        res = model.fit()

        ypred = res.predict(df_te)
        yte = df_te[feat].values
        test_r2s.append(r2_score(yte, ypred))

    glm_results.append({
        'feature': feat,
        'GLM_LONO_R2_with_interactions': np.mean(test_r2s)
    })

    # p-values from full-data fit
    model_full = smf.glm(formula=formula, data=df_z, family=sm.families.Gaussian())
    res_full = model_full.fit()
    pvals = {term: res_full.pvalues.get(term, np.nan) for term in interaction_terms}
    row = {'feature': feat}
    row.update(pvals)
    glm_pvals.append(row)

glm_df = pd.DataFrame(glm_results)
glm_df.to_excel("lsfm_GLM_LONO_glm_interactions_summary.xlsx", index=False)

glm_pval_df = pd.DataFrame(glm_pvals)
glm_pval_df.to_excel("para_location_interactions_pvals_GLM.xlsx", index=False)

# =========================
# Combine all results (Linear, Mixed, GLM)
# =========================
combined = linear_df.merge(mixed_df, on='feature').merge(glm_df, on='feature')
combined.to_excel("lsfm_GLM_LONO_combined_interactions_all.xlsx", index=False)

print("Done! Saved outputs:")
print("- lsfm_GLM_LONO_linear_interactions_summary.xlsx")
print("- lsfm_GLM_LONO_mixed_interactions_summary.xlsx")
print("- para_location_interactions_pvals.xlsx")
print("- lsfm_GLM_LONO_glm_interactions_summary.xlsx")
print("- lsfm_GLM_LONO_combined_interactions_all.xlsx")
