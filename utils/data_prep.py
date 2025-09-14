"""
This file hold:
 
    - All the pandas data preparation and preprocessing methods 
    - All the Test Methods of Hypothesis are defined here 

Sole purpose of this is to:
    - Select the correct data cleaning & preparing process for the Test
    - Select the correct method of Test     
    
"""


# import pandas as pd
# from scipy import stats
# from statsmodels.stats.weightstats import ztest
# import statsmodels.api as sm
# from typing import Dict, Any

# # =========================================================================
# # DATA PREPARATION METHODS
# # These methods handle the data preprocessing based on the LLM's parameters.
# # =========================================================================

# def prepare_data_for_ttest(df: pd.DataFrame, params: Dict[str, Any]) -> tuple:
#     """Prepares data for a two-sample t-test or Mann-Whitney U test."""
#     dependent_var = params['dependent_variable']
#     independent_var = params['independent_variables'][0]
#     sample_groups = params['sample_groups']

#     group1_df = df[df[independent_var] == sample_groups[0]][dependent_var].dropna()
#     group2_df = df[df[independent_var] == sample_groups[1]][dependent_var].dropna()

#     if group1_df.empty or group2_df.empty:
#         raise ValueError("One or both sample groups are empty after filtering.")

#     return group1_df, group2_df

# def prepare_data_for_anova(df: pd.DataFrame, params: Dict[str, Any]) -> list:
#     """Prepares data for an ANOVA or Kruskal-Wallis test."""
#     dependent_var = params['dependent_variable']
#     independent_var = params['independent_variables'][0]
    
#     # Get all unique groups in the specified column
#     groups = [group.dropna().values for name, group in df.groupby(independent_var)[dependent_var]]
    
#     if len(groups) < 2:
#         raise ValueError("ANOVA or Kruskal-Wallis requires at least two groups of data.")

#     return groups

# def prepare_data_for_chi_square(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
#     """Prepares a contingency table for Chi-Square or Fisher's test."""
#     col1 = params['columns'][0]
#     col2 = params['columns'][1]
    
#     return pd.crosstab(df[col1], df[col2])

# # =========================================================================
# # STATISTICAL TEST METHODS
# # These methods run the actual tests on the prepared data.
# # =========================================================================

# def run_one_sample_ttest(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
#     """Runs a one-sample t-test."""
#     dependent_var = params['dependent_variable']
#     pop_mean = params['population_mean']
#     tail = params['tail']
    
#     t_stat, p_value = stats.ttest_1samp(df[dependent_var].dropna(), pop_mean, alternative=tail)
    
#     return {"t_statistic": float(t_stat), "p_value": float(p_value)}

# def run_two_sample_ttest(groups: tuple, params: Dict[str, Any]) -> Dict[str, Any]:
#     """Runs an independent two-sample t-test on pre-grouped data."""
#     tail = params['tail']
#     equal_var = params['options'].get('equal_variance', True)
    
#     t_stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=equal_var, alternative=tail)
    
#     return {"t_statistic": float(t_stat), "p_value": float(p_value)}

# def run_paired_ttest(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
#     """Runs a paired t-test."""
#     # Paired tests don't have traditional dependent/independent vars, so they use the 'columns' key.
#     x_var, y_var = params['columns'][0], params['columns'][1]
#     tail = params['tail']
    
#     t_stat, p_value = stats.ttest_rel(df[x_var].dropna(), df[y_var].dropna(), alternative=tail)

#     return {"t_statistic": float(t_stat), "p_value": float(p_value)}

# def run_ztest(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
#     """Runs a one-sample z-test."""
#     dependent_var = params['dependent_variable']
#     pop_mean = params['population_mean']
#     pop_std = params.get('population_std', 'legacy') # 'legacy' is the default for a one-sample z-test
    
#     z_stat, p_value = ztest(x1=df[dependent_var].dropna(), value=pop_mean, alternative=params['tail'], ddof=pop_std)

#     return {"z_statistic": float(z_stat), "p_value": float(p_value)}

# def run_anova(groups: list, params: Dict[str, Any]) -> Dict[str, Any]:
#     """Runs a one-way ANOVA test on pre-grouped data."""
#     f_stat, p_value = stats.f_oneway(*groups)
    
#     return {"f_statistic": float(f_stat), "p_value": float(p_value)}

# def run_linear_regression(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
#     """Runs a linear regression analysis."""
#     dependent_var = params['dependent_variable']
#     independent_vars = params['independent_variables']

#     formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
#     model = sm.formula.ols(formula=formula, data=df.dropna()).fit()

#     return {
#         "summary": model.summary().as_html(),
#         "r_squared": float(model.rsquared),
#         "f_pvalue": float(model.f_pvalue)
#     }

# def run_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
#     """Runs a Pearson or Spearman correlation test."""
#     x_var = params['columns'][0]
#     y_var = params['columns'][1]
#     method = params['options'].get('method', 'pearson')
    
#     if method.lower() == 'pearson':
#         corr, p_value = stats.pearsonr(df[x_var].dropna(), df[y_var].dropna())
#     elif method.lower() == 'spearman':
#         corr, p_value = stats.spearmanr(df[x_var].dropna(), df[y_var].dropna())
#     else:
#         raise ValueError("Invalid correlation method specified.")

#     return {"correlation_coefficient": float(corr), "p_value": float(p_value)}

# def run_mann_whitney_u(groups: tuple, params: Dict[str, Any]) -> Dict[str, Any]:
#     """Runs a Mann-Whitney U test on pre-grouped data."""
#     tail = params['tail']
#     u_stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative=tail)
    
#     return {"u_statistic": float(u_stat), "p_value": float(p_value)}

# def run_wilcoxon(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
#     """Runs a Wilcoxon signed-rank test."""
#     # Wilcoxon is a paired test, so we look for columns in the 'columns' key.
#     x_var, y_var = params['columns'][0], params['columns'][1]
#     tail = params['tail']

#     t_stat, p_value = stats.wilcoxon(df[x_var].dropna(), df[y_var].dropna(), alternative=tail)
    
#     return {"t_statistic": float(t_stat), "p_value": float(p_value)}

# def run_kruskal(groups: list, params: Dict[str, Any]) -> Dict[str, Any]:
#     """Runs a Kruskal-Wallis H test on pre-grouped data."""
#     h_stat, p_value = stats.kruskal(*groups)
    
#     return {"h_statistic": float(h_stat), "p_value": float(p_value)}

# def run_chi_square_independence(contingency_table: pd.DataFrame) -> Dict[str, Any]:
#     """Runs a Chi-Square Test of Independence on a pre-built table."""
#     chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    
#     return {"chi2_statistic": float(chi2), "p_value": float(p_value)}

# def run_chi_square_goodness_of_fit(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
#     """Runs a Chi-Square Goodness-of-Fit Test."""
#     col = params['columns'][0]
#     observed_values = df[col].value_counts().sort_index().values
#     expected_values = params['expected_values']
    
#     chi2, p_value = stats.chisquare(f_obs=observed_values, f_exp=expected_values)
    
#     return {"chi2_statistic": float(chi2), "p_value": float(p_value)}
    
# def run_fishers_exact(contingency_table: pd.DataFrame) -> Dict[str, Any]:
#     """Runs a Fisher's Exact Test on a pre-built table."""
#     odds_ratio, p_value = stats.fisher_exact(contingency_table)
    
#     return {"odds_ratio": float(odds_ratio), "p_value": float(p_value)}



# ---------------------------------------------------------------------------
#  data_prep.py  –  updated with explicit numeric-cleaning + rich errors
# ---------------------------------------------------------------------------
import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import ztest
import statsmodels.api as sm
from typing import Dict, Any, List, Tuple

# ---------------------------------------------------------------------------
#  1.  ERROR-HELPER  (single-file version – move to separate module if desired)
# ---------------------------------------------------------------------------
def _fmt_error(func: str, col: str, value, hint: str = "") -> str:
    """Return a consistent, human-readable error string."""
    return (f"{func}: column '{col}' contains non-numeric value {value!r}. "
            f"{hint}").strip()

# ---------------------------------------------------------------------------
#  2.  GENERIC NUMERIC-CLEANER
# ---------------------------------------------------------------------------
def _ensure_numeric(series: pd.Series, func: str, col: str) -> pd.Series:
    """
    Coerce pd.Series to numeric.  If any original non-NA value becomes NA,
    raise immediately with a helpful message.
    """
    orig = series.copy()
    cleaned = pd.to_numeric(orig, errors="coerce")

    # locate first offending value
    bad_mask = orig.notna() & cleaned.isna()
    if bad_mask.any():
        first_bad = orig.loc[bad_mask].iloc[0]
        raise TypeError(_fmt_error(func, col, first_bad,
                                   "Remove or fix this value before proceeding."))
    return cleaned

# =========================================================================
# DATA-PREPARATION METHODS
# =========================================================================
def prepare_data_for_ttest(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
    """Prepares data for two-sample t-test or Mann-Whitney U test."""
    dep_var = params["dependent_variable"]
    indep_var = params["independent_variables"][0]
    groups = params["sample_groups"]  # list of two group labels

    # ensure numeric
    df = df.assign(**{dep_var: _ensure_numeric(df[dep_var], "prepare_data_for_ttest", dep_var)})

    g1 = df.loc[df[indep_var].eq(groups[0]), dep_var].dropna()
    g2 = df.loc[df[indep_var].eq(groups[1]), dep_var].dropna()

    if g1.empty or g2.empty:
        raise ValueError("One or both sample groups are empty after filtering.")

    return g1, g2

def prepare_data_for_anova(df: pd.DataFrame, params: Dict[str, Any]) -> List[pd.Series]:
    """Prepares data for ANOVA or Kruskal-Wallis test."""
    dep_var = params["dependent_variable"]
    indep_var = params["independent_variables"][0]

    # ensure numeric
    df = df.assign(**{dep_var: _ensure_numeric(df[dep_var], "prepare_data_for_anova", dep_var)})

    groups = [grp.dropna() for _, grp in df.groupby(indep_var)[dep_var]]
    groups = [g for g in groups if not g.empty]

    if len(groups) < 2:
        raise ValueError("ANOVA/Kruskal-Wallis requires at least two non-empty groups.")

    return groups

def prepare_data_for_chi_square(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Builds a contingency table for Chi-Square or Fisher's exact test."""
    col1, col2 = params["columns"][0], params["columns"][1]
    return pd.crosstab(df[col1], df[col2])

# =========================================================================
# STATISTICAL TEST RUNNERS
# =========================================================================
def run_one_sample_ttest(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    dep_var = params["dependent_variable"]
    pop_mean = params["population_mean"]
    tail = params["tail"]

    cleaned = _ensure_numeric(df[dep_var], "run_one_sample_ttest", dep_var)
    t_stat, p_val = stats.ttest_1samp(cleaned.dropna(), pop_mean, alternative=tail)
    return {"t_statistic": float(t_stat), "p_value": float(p_val)}

def run_two_sample_ttest(groups: Tuple[pd.Series, pd.Series], params: Dict[str, Any]) -> Dict[str, Any]:
    tail = params["tail"]
    equal_var = params["options"].get("equal_variance", True)
    t_stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=equal_var, alternative=tail)
    return {"t_statistic": float(t_stat), "p_value": float(p_val)}

def run_paired_ttest(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    x_var, y_var = params["columns"][0], params["columns"][1]
    tail = params['test_parameters']["tail"]
    x_clean = _ensure_numeric(df[x_var], "run_paired_ttest", x_var)
    y_clean = _ensure_numeric(df[y_var], "run_paired_ttest", y_var)
    t_stat, p_val = stats.ttest_rel(x_clean.dropna(), y_clean.dropna(), alternative=tail)
    return {"t_statistic": float(t_stat), "p_value": float(p_val)}

def run_ztest(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    dep_var = params["dependent_variable"]
    pop_mean = params["population_mean"]
    pop_std = params.get("population_std", "legacy")  # statsmodels default
    cleaned = _ensure_numeric(df[dep_var], "run_ztest", dep_var)
    z_stat, p_val = ztest(cleaned.dropna(), value=pop_mean, alternative=params["tail"], ddof=pop_std)
    return {"z_statistic": float(z_stat), "p_value": float(p_val)}

def run_anova(groups: List[pd.Series]) -> Dict[str, Any]:
    f_stat, p_val = stats.f_oneway(*groups)
    return {"f_statistic": float(f_stat), "p_value": float(p_val)}

def run_linear_regression(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    dep_var = params["dependent_variable"]
    indep_vars = params["independent_variables"]
    formula = f"{dep_var} ~ {' + '.join(indep_vars)}"
    model = sm.formula.ols(formula, data=df.dropna()).fit()
    return {
        "summary": model.summary().as_html(),
        "r_squared": float(model.rsquared),
        "f_pvalue": float(model.f_pvalue),
    }

def run_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    x_var, y_var = params["columns"][0], params["columns"][1]
    print(f"x_var: {x_var}")
    method = params["test_parameters"]["options"].get("method", "pearson").lower()
    x_clean = _ensure_numeric(df[x_var], "run_correlation", x_var)
    y_clean = _ensure_numeric(df[y_var], "run_correlation", y_var)
    if method == "pearson":
        corr, p_val = stats.pearsonr(x_clean.dropna(), y_clean.dropna())
    elif method == "spearman":
        corr, p_val = stats.spearmanr(x_clean.dropna(), y_clean.dropna())
    else:
        raise ValueError("Invalid correlation method specified.")
    return {"correlation_coefficient": float(corr), "p_value": float(p_val)}

def run_mann_whitney_u(groups: Tuple[pd.Series, pd.Series], params: Dict[str, Any]) -> Dict[str, Any]:
    tail = params["tail"]
    u_stat, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative=tail)
    return {"u_statistic": float(u_stat), "p_value": float(p_val)}

def run_wilcoxon(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    x_var, y_var = params["columns"][0], params["columns"][1]
    tail = params["test_parameters"]["tail"]
    x_clean = _ensure_numeric(df[x_var], "run_wilcoxon", x_var)
    y_clean = _ensure_numeric(df[y_var], "run_wilcoxon", y_var)
    t_stat, p_val = stats.wilcoxon(x_clean.dropna(), y_clean.dropna(), alternative=tail)
    return {"t_statistic": float(t_stat), "p_value": float(p_val)}

def run_kruskal(groups: List[pd.Series], params: Dict[str, Any]) -> Dict[str, Any]:
    h_stat, p_val = stats.kruskal(*groups)
    return {"h_statistic": float(h_stat), "p_value": float(p_val)}

def run_chi_square_independence(contingency_table: pd.DataFrame) -> Dict[str, Any]:
    chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
    return {"chi2_statistic": float(chi2), "p_value": float(p_val)}

def run_chi_square_goodness_of_fit(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    col = params["columns"][0]
    observed = df[col].value_counts().sort_index().values
    expected = params["test_parameters"]["expected_values"]
    chi2, p_val = stats.chisquare(f_obs=observed, f_exp=expected)
    return {"chi2_statistic": float(chi2), "p_value": float(p_val)}

def run_fishers_exact(contingency_table: pd.DataFrame) -> Dict[str, Any]:
    odds_ratio, p_val = stats.fisher_exact(contingency_table)
    return {"odds_ratio": float(odds_ratio), "p_value": float(p_val)}