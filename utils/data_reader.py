"""
This file is repsonsible for following things:

1 - This will be called first in the handle() method under agent_layer.py 

2 - This one gets into work once the frontend sends back the dataframe, columns and user_prompt to handle() method

3 - It reads the dataframe and makes it more comprehensive for the LLM by extracting values like: 

    - Indentifies the Numerical & Categorical columns in passed column from Frontend User Input
    - Normality Test : Runs a Shapiro-Wilk or KS-Test test on the numeric columns(if present in user given columns) to check if data is from normal distribution or not. This is a key assumption for parametric tests like the t-test and ANOVA.
    - Homoscedasiticity Test : Runs a levene test if, the given columns have exactly one numerical and one categorical column

    - It also gives count of total_rows, missing values, count of unique values in categorical data, statistical description of numerical columns(if any)    
"""


import pandas as pd
from scipy.stats import shapiro, levene, kstest
import numpy as np
import json

def prepare_data_context(df: pd.DataFrame, columns: list[str]):

    context = {}

    # 1. Add overall DataFrame info
    context["total_rows"] = len(df)
    context["columns_list"] = columns
    
    # 2. Add column-specific details with a flat naming convention
    for col in columns:
        col_type = str(df[col].dtype)
        
        context[f"{col}_dtype"] = col_type
        context[f"{col}_missing_count"] = int(df[col].isnull().sum())
        
        if np.issubdtype(df[col].dtype, np.number):
            describe_series = df[col].describe()
            for metric in describe_series.index:
                context[f"{col}_{metric}"] = float(describe_series[metric])
            
            # --- Adaptive Normality Test Logic ---
            valid_data = df[col].dropna()
            
            # Check if there's enough data and if it's too large for Shapiro-Wilk
            if len(valid_data) > 3 and len(valid_data) <= 5000:
                try:
                    normality_test_pvalue = shapiro(valid_data).pvalue
                    context[f"{col}_normality_test_pvalue"] = np.round(float(normality_test_pvalue), 4)
                    context[f"{col}_normality_test_used"] = "shapiro"
                except:
                    context[f"{col}_normality_test_pvalue"] = None
                    context[f"{col}_normality_test_used"] = None
            elif len(valid_data) > 5000:
                try:
                    normality_test_pvalue = kstest(valid_data, 'norm').pvalue
                    context[f"{col}_normality_test_pvalue"] = np.round(float(normality_test_pvalue), 4)
                    context[f"{col}_normality_test_used"] = "kstest"
                except:
                    context[f"{col}_normality_test_pvalue"] = None
                    context[f"{col}_normality_test_used"] = None
            else:
                 context[f"{col}_normality_test_pvalue"] = None
                 context[f"{col}_normality_test_used"] = None
            # --- End of adaptive test logic ---
            
        else:
            unique_values = df[col].dropna().unique().tolist()
            context[f"{col}_unique_count"] = len(unique_values)
            context[f"{col}_unique_values"] = unique_values
            
    # 3. Add inter-column relationship tests (like Levene's)
    numeric_cols = [col for col in columns if np.issubdtype(df[col].dtype, np.number)]
    categorical_cols = [col for col in columns if not np.issubdtype(df[col].dtype, np.number)]
    
    if len(numeric_cols) == 1 and len(categorical_cols) >= 1:
        dependent_var = numeric_cols[0]
        grouping_var = categorical_cols[0]
        
        try:
            groups = [group.dropna().values for name, group in df[[dependent_var, grouping_var]].groupby(grouping_var)[dependent_var]]
            
            if len(groups) >= 2:
                levene_pvalue = levene(*groups).pvalue
                context[f"{dependent_var}_levene_pvalue"] = np.round(float(levene_pvalue), 4)
            else:
                context[f"{dependent_var}_levene_pvalue"] = None
        except Exception as e:
            context[f"{dependent_var}_levene_pvalue"] = None
            
    # Return as a string for LLM, converting to JSON first for proper formatting
   
    return json.dumps(context)