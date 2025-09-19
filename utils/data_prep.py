"""
This file hold:
 
    - All the pandas data preparation and preprocessing methods 
    - All the Test Methods of Hypothesis are defined here 

Sole purpose of this is to:
    - Select the correct data cleaning & preparing process for the Test
    - Select the correct method of Test     
    
"""

# ---------------------------------------------------------------------------
#  data_prep.py  –  updated with explicit numeric-cleaning + rich errors
# ---------------------------------------------------------------------------
import pandas as pd
from scipy import stats
import numpy as np
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
    
    tail = ''
    if params.get('tail', None) == 'two-tailed':
        tail = 'two-sided'
    elif params.get('tail', None) == 'left-tailed':
        tail = 'less'
    elif params.get('tail', None) == 'right-tailed':
        tail = 'greater'

    cleaned = _ensure_numeric(df[dep_var], "run_one_sample_ttest", dep_var)
    
    
    n = len(cleaned.dropna())   # sample size

    t_stat, p_val = stats.ttest_1samp(cleaned.dropna(), pop_mean, alternative=tail)
    return {"t_statistic": float(t_stat), "p_value": float(p_val), "degrees_of_freedom": n-1}

def run_two_sample_ttest(groups: Tuple[pd.Series, pd.Series], params: Dict[str, Any]) -> Dict[str, Any]:
    tail = ''

    if params["tail"] == 'two-tailed':
        tail = 'two-sided'
    elif params["tail"] == 'left-tailed':
        tail = 'less'
    elif params["tail"] == 'right-tailed':
        tail = 'greater'

    equal_var = params["options"].get("equal_variance", True)
    t_stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=equal_var, alternative=tail)
    n1 = len(groups[0])
    n2 = len(groups[1])
    df_val = n1 + n2 - 2
    return {"t_statistic": float(t_stat), "p_value": float(p_val), "degrees_of_freedom": df_val}

def run_paired_ttest(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    x_var, y_var = params["columns"][0], params["columns"][1]

    tail = ''
    if params.get('test_parameters', {}).get('tail', None) == 'two-tailed':
        tail = 'two-sided'
    elif params.get('test_parameters', {}).get('tail', None) == 'left-tailed':
        tail = 'less'
    elif params.get('test_parameters', {}).get('tail', None) == 'right-tailed':
        tail = 'greater'
    
    x_clean = _ensure_numeric(df[x_var], "run_paired_ttest", x_var)
    y_clean = _ensure_numeric(df[y_var], "run_paired_ttest", y_var)

    paired_df = pd.DataFrame({x_var: x_clean, y_var: y_clean}).dropna()
    n = len(paired_df)

    t_stat, p_val = stats.ttest_rel(x_clean.dropna(), y_clean.dropna(), alternative=tail)

    return {"t_statistic": float(t_stat), "p_value": float(p_val), "degrees_of_freedom": n-1}

def run_ztest(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    dep_var = params.get("dependent_variable")
    pop_mean = params.get("population_mean")
    pop_std = params.get("population_std")  # must be provided

    cleaned = _ensure_numeric(df[dep_var], "run_ztest", dep_var).dropna()
    n = len(cleaned)
    sample_mean = cleaned.mean()

    # Compute z-statistic
    z_stat = (sample_mean - pop_mean) / (pop_std / np.sqrt(n))

    # Tail handling
    tail = params.get("tail", "two-tailed")
    if tail == "two-tailed":
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif tail == "left-tailed":
        p_val = stats.norm.cdf(z_stat)
    else:  # right-tailed
        p_val = 1 - stats.norm.cdf(z_stat)

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

# def run_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
#     x_var, y_var = params["columns"][0], params["columns"][1]
#     print(f"x_var: {x_var}")
#     method = params["test_parameters"]["options"].get("method", "pearson").lower()
#     x_clean = _ensure_numeric(df[x_var], "run_correlation", x_var)
#     y_clean = _ensure_numeric(df[y_var], "run_correlation", y_var)
#     if method == "pearson":
#         corr, p_val = stats.pearsonr(x_clean.dropna(), y_clean.dropna())
#     elif method == "spearman":
#         corr, p_val = stats.spearmanr(x_clean.dropna(), y_clean.dropna())
#     else:
#         raise ValueError("Invalid correlation method specified.")
#     return {"correlation_coefficient": float(corr), "p_value": float(p_val)}

def run_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs a Pearson or Spearman correlation test.
    If an ordinal_mapping is provided in the parameters, it will be used to
    encode categorical data into numeric ranks before running the test.
    """
    x_var, y_var = params["columns"][0], params["columns"][1]
    method = params["test_parameters"]["options"].get("method", "pearson").lower()
    
    # Safely get the ordinal mapping from the parameters
    ordinal_mapping = params["test_parameters"].get("ordinal_mapping")

    # Start with copies of the original data series
    x_series = df[x_var].copy()
    y_series = df[y_var].copy()

    # --- NEW LOGIC to apply ordinal encoding ---
    if ordinal_mapping:
        print(f"Ordinal mapping found: {ordinal_mapping}. Applying to data.")
        # The mapping is a dict like {'column_name': {'category': 1, ...}}
        for col_to_map, mapping_dict in ordinal_mapping.items():
            if col_to_map == x_var:
                print(f"Mapping column '{x_var}'")
                x_series = x_series.map(mapping_dict)
            elif col_to_map == y_var:
                print(f"Mapping column '{y_var}'")
                y_series = y_series.map(mapping_dict)
    
    # Now, with ordinal data encoded, we can proceed to ensure everything is numeric
    # This will catch any remaining non-numeric issues
    x_clean = _ensure_numeric(x_series, "run_correlation", x_var)
    y_clean = _ensure_numeric(y_series, "run_correlation", y_var)

    # --- IMPROVED LOGIC for handling missing values ---
    # Combine into a single DataFrame to ensure rows are aligned, then drop NaNs
    combined_df = pd.DataFrame({'x': x_clean, 'y': y_clean}).dropna()

    # Check if there's enough data left to perform a correlation
    if len(combined_df) < 2:
        return {
            "error": "Not enough overlapping non-null data points to calculate a correlation."
        }
        
    # Select the method and run the test on the cleaned, aligned data
    if method == "pearson":
        corr, p_val = stats.pearsonr(combined_df['x'], combined_df['y'])
    elif method == "spearman":
        corr, p_val = stats.spearmanr(combined_df['x'], combined_df['y'])
    else:
        raise ValueError(f"Invalid correlation method specified: '{method}'")

    return {
        "correlation_coefficient": float(corr),
        "p_value": float(p_val),
        # "n_observations": len(combined_df) # It's good practice to return the sample size
    }


def run_mann_whitney_u(groups: Tuple[pd.Series, pd.Series], params: Dict[str, Any]) -> Dict[str, Any]:
    tail = ''
    if params["tail"] == 'two-tailed':
        tail = 'two-sided'
    elif params["tail"] == 'left-tailed':
        tail = 'less'
    elif params["tail"] == 'right-tailed':
        tail = 'greater'
    u_stat, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative=tail)
    return {"u_statistic": float(u_stat), "p_value": float(p_val)}

def run_wilcoxon(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    x_var, y_var = params["columns"][0], params["columns"][1]
    
    tail = ''
    if params.get('test_parameters', {}).get('tail', None) == 'two-tailed':
        tail = 'two-sided'
    elif params.get('test_parameters', {}).get('tail', None) == 'left-tailed':
        tail = 'less'
    elif params.get('test_parameters', {}).get('tail', None) == 'right-tailed':
        tail = 'greater'

    x_clean = _ensure_numeric(df[x_var], "run_wilcoxon", x_var)
    y_clean = _ensure_numeric(df[y_var], "run_wilcoxon", y_var)
    t_stat, p_val = stats.wilcoxon(x_clean.dropna(), y_clean.dropna(), alternative=tail)

    n = len(x_var)
    w = (t_stat - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24) # for plotting, turning the t_stat into zstat
    
    return {"t_statistic": float(w), "p_value": float(p_val)}

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