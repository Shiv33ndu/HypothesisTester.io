import numpy as np
import pandas as pd
from scipy import stats

# --------------------------
# 1. One-Sample t-test
# --------------------------
def one_sample_ttest(data: pd.Series, population_mean: float, tail: str = "two-sided"):
    """
    One-sample t-test: test if mean of sample differs from population mean.
    tail: 'two-sided', 'greater', 'less'
    """
    t_stat, p_value = stats.ttest_1samp(data, population_mean)

    # Adjust p-value for one-tailed tests
    if tail == "greater":
        p_value = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)
    elif tail == "less":
        p_value = p_value / 2 if t_stat < 0 else 1 - (p_value / 2)

    return {"t_stat": t_stat, "p_value": p_value, "tail": tail}


# --------------------------
# 2. Two-Sample t-test
# --------------------------
def two_sample_ttest(group1: pd.Series, group2: pd.Series, tail: str = "two-sided", equal_var: bool = False):
    """
    Independent two-sample t-test: compare means of two groups.
    tail: 'two-sided', 'greater', 'less'
    """
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)

    if tail == "greater":
        p_value = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)
    elif tail == "less":
        p_value = p_value / 2 if t_stat < 0 else 1 - (p_value / 2)

    return {"t_stat": t_stat, "p_value": p_value, "tail": tail}


# --------------------------
# 3. Paired t-test
# --------------------------
def paired_ttest(before: pd.Series, after: pd.Series, tail: str = "two-sided"):
    """
    Paired t-test: compare means of two related samples (e.g. before vs after).
    """
    t_stat, p_value = stats.ttest_rel(before, after)

    if tail == "greater":
        p_value = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)
    elif tail == "less":
        p_value = p_value / 2 if t_stat < 0 else 1 - (p_value / 2)

    return {"t_stat": t_stat, "p_value": p_value, "tail": tail}


# --------------------------
# 4. ANOVA
# --------------------------
def anova(*groups):
    """
    One-way ANOVA: compare means across 2+ groups.
    """
    f_stat, p_value = stats.f_oneway(*groups)
    return {"f_stat": f_stat, "p_value": p_value}


# --------------------------
# 5. Chi-Square Test of Independence
# --------------------------
def chi_square_test(col1: pd.Series, col2: pd.Series):
    """
    Chi-square test for independence between two categorical variables.
    """
    contingency_table = pd.crosstab(col1, col2)
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    return {"chi2": chi2, "p_value": p_value, "dof": dof, "expected": expected}


# --------------------------
# 6. Correlation Test
# --------------------------
def correlation_test(x: pd.Series, y: pd.Series, method: str = "pearson"):
    """
    Correlation test: Pearson (default), Spearman, or Kendall.
    """
    if method == "pearson":
        corr, p_value = stats.pearsonr(x, y)
    elif method == "spearman":
        corr, p_value = stats.spearmanr(x, y)
    elif method == "kendall":
        corr, p_value = stats.kendalltau(x, y)
    else:
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'")
    
    return {"correlation": corr, "p_value": p_value, "method": method}
