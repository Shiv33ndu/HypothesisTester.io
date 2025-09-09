import numpy as np
import pandas as pd
from typing import Dict, Any

from utils import data_prep 

# =========================================================================
# CENTRAL TEST DISPATCHER
# This is the main function that coordinates everything from data_prep.py
# =========================================================================

def dispatch_test(llm_output_json: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Acts as the central dispatcher for all hypothesis tests.
    It takes the LLM's plan and executes the correct test method.
    """
    test_name = llm_output_json['test_name']
    params = llm_output_json['test_parameters']
    
    print(f"test_name : {test_name}")

    try:
        # Step 1: Data Preparation based on test type
        if test_name in ["Two-Sample Independent t-test", "Mann-Whitney U Test"]:
            prepared_data = data_prep.prepare_data_for_ttest(df, params)
            test_results = data_prep.run_two_sample_ttest(prepared_data, params) if test_name == "Two-Sample Independent t-test" else data_prep.run_mann_whitney_u(prepared_data, params)
        elif test_name in ["ANOVA", "Kruskal-Wallis H Test"]:
            prepared_data = data_prep.prepare_data_for_anova(df, params)
            test_results = data_prep.run_anova(prepared_data, params) if test_name == "ANOVA" else data_prep.run_kruskal(prepared_data, params)
        elif test_name in ["Chi-Square Test of Independence", "Fisher's Exact Test"]:
            prepared_data = data_prep.prepare_data_for_chi_square(df, params)
            test_results = data_prep.run_chi_square_independence(prepared_data) if test_name == "Chi-Square Test of Independence" else data_prep.run_fishers_exact(prepared_data)
        else:
            # For tests that don't need complex data prep, run them directly
            if test_name == "One-Sample t-test":
                test_results = data_prep.run_one_sample_ttest(df, params)
            elif test_name == "Paired t-test":
                test_results = data_prep.run_paired_ttest(df, params)
            elif test_name == "One-Sample Z-test":
                test_results = data_prep.run_ztest(df, params)
            elif test_name == "Linear Regression Analysis":
                test_results = data_prep.run_linear_regression(df, params)
            elif test_name == "Correlation Test (Pearson/Spearman)":
                test_results = data_prep.run_correlation(df, params)
            elif test_name == "Wilcoxon Signed-Rank Test":
                test_results = data_prep.run_wilcoxon(df, params)
            elif test_name == "Chi-Square Goodness-of-Fit Test":
                test_results = data_prep.run_chi_square_goodness_of_fit(df, params)
            else:
                return {"error": f"Test '{test_name}' is not a recognized or implemented test."}

        # Step 3: Return the results in a final, clean format
        
        print('in hypothesis_test')
        print(llm_output_json)
        
        return llm_output_json, test_results

    except Exception as e:
        return {"error": f"An error occurred during test execution: {str(e)}"}


