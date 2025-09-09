"""
This file hold:
 
    - All the pandas data preparation and preprocessing methods 
    - All the Test Methods of Hypothesis are defined here 

Sole purpose of this is to:
    - Select the correct data cleaning & preparing process for the Test
    - Select the correct method of Test     
    
"""


import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import ztest
import statsmodels.api as sm
from typing import Dict, Any

# =========================================================================
# DATA PREPARATION METHODS
# These methods handle the data preprocessing based on the LLM's parameters.
# =========================================================================

def prepare_data_for_ttest(df: pd.DataFrame, params: Dict[str, Any]) -> tuple:
    """Prepares data for a two-sample t-test or Mann-Whitney U test."""
    dependent_var = params['dependent_variable']
    independent_var = params['independent_variables'][0]
    sample_groups = params['sample_groups']

    group1_df = df[df[independent_var] == sample_groups[0]][dependent_var].dropna()
    group2_df = df[df[independent_var] == sample_groups[1]][dependent_var].dropna()

    if group1_df.empty or group2_df.empty:
        raise ValueError("One or both sample groups are empty after filtering.")

    return group1_df, group2_df

def prepare_data_for_anova(df: pd.DataFrame, params: Dict[str, Any]) -> list:
    """Prepares data for an ANOVA or Kruskal-Wallis test."""
    dependent_var = params['dependent_variable']
    independent_var = params['independent_variables'][0]
    
    # Get all unique groups in the specified column
    groups = [group.dropna().values for name, group in df.groupby(independent_var)[dependent_var]]
    
    if len(groups) < 2:
        raise ValueError("ANOVA or Kruskal-Wallis requires at least two groups of data.")

    return groups

def prepare_data_for_chi_square(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Prepares a contingency table for Chi-Square or Fisher's test."""
    col1 = params['columns'][0]
    col2 = params['columns'][1]
    
    return pd.crosstab(df[col1], df[col2])

# =========================================================================
# STATISTICAL TEST METHODS
# These methods run the actual tests on the prepared data.
# =========================================================================

def run_one_sample_ttest(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a one-sample t-test."""
    dependent_var = params['dependent_variable']
    pop_mean = params['population_mean']
    tail = params['tail']
    
    t_stat, p_value = stats.ttest_1samp(df[dependent_var].dropna(), pop_mean, alternative=tail)
    
    return {"t_statistic": float(t_stat), "p_value": float(p_value)}

def run_two_sample_ttest(groups: tuple, params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs an independent two-sample t-test on pre-grouped data."""
    tail = params['tail']
    equal_var = params['options'].get('equal_variance', True)
    
    t_stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=equal_var, alternative=tail)
    
    return {"t_statistic": float(t_stat), "p_value": float(p_value)}

def run_paired_ttest(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a paired t-test."""
    # Paired tests don't have traditional dependent/independent vars, so they use the 'columns' key.
    x_var, y_var = params['columns'][0], params['columns'][1]
    tail = params['tail']
    
    t_stat, p_value = stats.ttest_rel(df[x_var].dropna(), df[y_var].dropna(), alternative=tail)

    return {"t_statistic": float(t_stat), "p_value": float(p_value)}

def run_ztest(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a one-sample z-test."""
    dependent_var = params['dependent_variable']
    pop_mean = params['population_mean']
    pop_std = params.get('population_std', 'legacy') # 'legacy' is the default for a one-sample z-test
    
    z_stat, p_value = ztest(x1=df[dependent_var].dropna(), value=pop_mean, alternative=params['tail'], ddof=pop_std)

    return {"z_statistic": float(z_stat), "p_value": float(p_value)}

def run_anova(groups: list, params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a one-way ANOVA test on pre-grouped data."""
    f_stat, p_value = stats.f_oneway(*groups)
    
    return {"f_statistic": float(f_stat), "p_value": float(p_value)}

def run_linear_regression(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a linear regression analysis."""
    dependent_var = params['dependent_variable']
    independent_vars = params['independent_variables']

    formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
    model = sm.formula.ols(formula=formula, data=df.dropna()).fit()

    return {
        "summary": model.summary().as_html(),
        "r_squared": float(model.rsquared),
        "f_pvalue": float(model.f_pvalue)
    }

def run_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a Pearson or Spearman correlation test."""
    x_var = params['columns'][0]
    y_var = params['columns'][1]
    method = params['options'].get('method', 'pearson')
    
    if method.lower() == 'pearson':
        corr, p_value = stats.pearsonr(df[x_var].dropna(), df[y_var].dropna())
    elif method.lower() == 'spearman':
        corr, p_value = stats.spearmanr(df[x_var].dropna(), df[y_var].dropna())
    else:
        raise ValueError("Invalid correlation method specified.")

    return {"correlation_coefficient": float(corr), "p_value": float(p_value)}

def run_mann_whitney_u(groups: tuple, params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a Mann-Whitney U test on pre-grouped data."""
    tail = params['tail']
    u_stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative=tail)
    
    return {"u_statistic": float(u_stat), "p_value": float(p_value)}

def run_wilcoxon(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a Wilcoxon signed-rank test."""
    # Wilcoxon is a paired test, so we look for columns in the 'columns' key.
    x_var, y_var = params['columns'][0], params['columns'][1]
    tail = params['tail']

    t_stat, p_value = stats.wilcoxon(df[x_var].dropna(), df[y_var].dropna(), alternative=tail)
    
    return {"t_statistic": float(t_stat), "p_value": float(p_value)}

def run_kruskal(groups: list, params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a Kruskal-Wallis H test on pre-grouped data."""
    h_stat, p_value = stats.kruskal(*groups)
    
    return {"h_statistic": float(h_stat), "p_value": float(p_value)}

def run_chi_square_independence(contingency_table: pd.DataFrame) -> Dict[str, Any]:
    """Runs a Chi-Square Test of Independence on a pre-built table."""
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    
    return {"chi2_statistic": float(chi2), "p_value": float(p_value)}

def run_chi_square_goodness_of_fit(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a Chi-Square Goodness-of-Fit Test."""
    col = params['columns'][0]
    observed_values = df[col].value_counts().sort_index().values
    expected_values = params['expected_values']
    
    chi2, p_value = stats.chisquare(f_obs=observed_values, f_exp=expected_values)
    
    return {"chi2_statistic": float(chi2), "p_value": float(p_value)}
    
def run_fishers_exact(contingency_table: pd.DataFrame) -> Dict[str, Any]:
    """Runs a Fisher's Exact Test on a pre-built table."""
    odds_ratio, p_value = stats.fisher_exact(contingency_table)
    
    return {"odds_ratio": float(odds_ratio), "p_value": float(p_value)}



