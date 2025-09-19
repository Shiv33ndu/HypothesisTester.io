import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from rapidfuzz import fuzz
import re
from modules.agent_layer import handle
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import t, norm

from modules.agent_layer import check_user_question
from modules.agent_layer import summarize

from langchain_core.messages import HumanMessage, AIMessage

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Hypothesis Tester AI", layout="wide")

st.title("🤖 Hypothesis Tester AI")

# -------------------------
# Custom sidebar width
# -------------------------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 30% !important;
    }
    [data-testid="stSidebarContent"] {
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Track app state
# -------------------------
if "ran_hypothesis" not in st.session_state:
    st.session_state.ran_hypothesis = False
if "results" not in st.session_state:
    st.session_state.results = None
if 'df' not in st.session_state:
    st.session_state.df = None

if 'hypotheses_quest' not in st.session_state:
    st.session_state.hypotheses_quest = None

if 'plotContext' not in st.session_state:
    st.session_state.plotContext = None

st.session_state.initialized = True

# ------------------------------------
# Column extraction from parenthesis
# ------------------------------------
def extract_from_parentheses(text):
    """
    Extracts all text enclosed in parentheses from a string.
    Returns a list of the found strings.
    """
    return re.findall(r'\((.*?)\)', text)


# -------------------------
# Sidebar UI
# -------------------------

with st.sidebar:
    st.title("⚙️ Controls")

    
    uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is None:
        st.session_state.ran_hypothesis = False

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write("📊 Preview:", st.session_state.df.head())
        df = st.session_state.df
        
        # Separate column types
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

        # User question
        st.subheader("💬 Ask Your Question")
        user_prompt = st.text_area(
            "Enter your hypothesis in plain English,. Hint: using column names in parenthesis() will give better results",
            placeholder="Is benefit score(benefits_score) negatively correlated with salary(salary_usd)?"
            )
        st.session_state.hypotheses_quest = user_prompt

        # Run Hypothesis
        if st.button("🔎 Run Hypothesis Test"):
            if not user_prompt:
                st.warning("Please enter a question before running!")
            
            # New logic to handle user-enforced format
            enclosed_cols = extract_from_parentheses(user_prompt)
            
            # Check if the user used the recommended parenthesis format
            if enclosed_cols:
                # User provided column names, let's validate them
                matched_cols = [col for col in enclosed_cols if col in df.columns]
                
                # Check for typos
                if len(matched_cols) != len(enclosed_cols):
                    st.error("⚠️ Column name mismatch! One or more of the columns you entered do not exist in the dataset. Please check for typos and try again.")

                    # 1. Identify the columns with typos
                    typo_cols = [col for col in enclosed_cols if col not in df.columns]
                    
                    # Generate the narrowed, fuzzy-matched hints
                    fuzzy_matches = []
                    for user_col in typo_cols:
                        # Find the best fuzzy match for this specific column name
                        best_match = None
                        highest_ratio = 0
                        
                        for df_col in df.columns:
                            ratio = fuzz.ratio(user_col.lower(), df_col.lower())
                            if ratio > highest_ratio:
                                highest_ratio = ratio
                                best_match = df_col
                        
                        # Add the best match if it's above a certain threshold (e.g., 70)
                        if highest_ratio >= 70 and best_match not in fuzzy_matches:
                            fuzzy_matches.append(f"{user_col} -> {best_match}")

                    st.error(f"Wrong column name you entered : {typo_cols} ")

                    if fuzzy_matches:
                        st.info(f"Did you mean: {', '.join(fuzzy_matches)} ?")

                    st.stop()
                
                # If all columns match, proceed with the validated list
                st.success(f"Columns found from your input : {matched_cols}")
                info_placeholder = st.empty()
                info_placeholder.info('🔍  Choosing the right test …')
                
                # moving forward to LLM layer to pass these values to work with and get the suggested test
                check_question = check_user_question(user_prompt, df) 
                
                info_placeholder.empty()          # remove old banner

                if check_question["decision"] == "descriptive":
                    with info_placeholder.container():
                        st.warning("⚠️  Not a hypothesis-testing question.")
                        st.markdown(
                            f"*Descriptive answer:*  \n"
                            f"**{check_question['answer']}**"
                        )
                        with st.expander("Why no test was run"):
                            st.write(check_question["reason"])
                
                else:
                    data_context_json, response, test_results = handle(user_prompt, df, matched_cols)  
                
                    # setting a dummy result as of now to show on the plot space
                    st.session_state.results = {"cols": matched_cols, "prompt": user_prompt, 'response' : response, 'test_results' : test_results, "data_context": data_context_json}
                    st.session_state.ran_hypothesis = True

            # Fallback to fuzzy search if no parentheses were used
            else:
                # extracting the column names using fuzzy search
                matched_cols = [
                    col for col in df.columns
                    if fuzz.partial_ratio(col.lower(), user_prompt.lower()) >= 70
                ]
                if matched_cols:
                    st.success(f"Inferred Columns : {matched_cols}")
                    
                    info_placeholder = st.empty()

                    info_placeholder.info('🔍  Choosing the right test …')

                    # passing the user prompt, and dataframe to LLM to identify correct Test for this 
                    check_question = check_user_question(user_prompt, df) 
                
                    info_placeholder.empty()          # remove old banner

                    if check_question.get("decision", "descriptive") == "descriptive":
                        with info_placeholder.container():
                            st.warning("⚠️  Not a hypothesis-testing question.")
                            st.markdown(
                                f"*Descriptive answer:*  \n"
                                f"**{check_question.get('answer', None)}**"
                            )
                            with st.expander("Why no test was run"):
                                st.write(check_question("reason", None))

                    else:
                        data_context_json, response, test_results = handle(user_prompt, df, matched_cols)  
                    
                        # setting a dummy result as of now to show on the plot space
                        st.session_state.results = {"cols": matched_cols, "prompt": user_prompt, 'response' : response, 'test_results' : test_results, "data_context": data_context_json}
                    
                        st.session_state.ran_hypothesis = True

                else:
                    st.error("⚠️ Could not infer columns. Try putting the relevant columns into parenthesis(column_name).")
                    st.stop()

       
        
# =========================================================================
# PLOTTING FUNCTIONS
# =========================================================================

def plot_ttest_significance(test_stat, p_value, df_val, tail_type, title, alpha=0.05):
    """
    Creates a visually appealing plot of the t-distribution, highlighting the
    rejection (red) and acceptance (green) regions.
    """
    x_values = np.linspace(-5, 5, 1000)
    y_values = t.pdf(x_values, df_val)
    
    fig = go.Figure()
    
    # Plot the main t-distribution curve with a fill
    fig.add_trace(go.Scatter(
        x=x_values, y=y_values, mode='lines', 
        fill='tozeroy', line_color='rgba(76, 114, 176, 1)', fillcolor='rgba(76, 114, 176, 0.2)',
        name='t-Distribution'
    ))
    
    # Determine significance and color for the test statistic line
    is_significant = p_value <= alpha
    stat_line_color = "red" if is_significant else "green"

    # Define critical values and shade regions
    if tail_type == "two-sided":
        crit_val_upper = t.ppf(1 - alpha/2, df_val)
        crit_val_lower = -crit_val_upper
        # Red rejection regions
        fig.add_vrect(x0=crit_val_upper, x1=5, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Rejection Region", annotation_position="top left")
        fig.add_vrect(x0=-5, x1=crit_val_lower, fillcolor="red", opacity=0.2, line_width=0)
        # Green "Fail to Reject" region
        fig.add_vrect(x0=crit_val_lower, x1=crit_val_upper, fillcolor="green", opacity=0.15, line_width=0, annotation_text="Fail to Reject Region", annotation_position="top")

    elif tail_type == "less":
        crit_val = t.ppf(alpha, df_val)
        fig.add_vrect(x0=-5, x1=crit_val, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Rejection Region")
        fig.add_vrect(x0=crit_val, x1=5, fillcolor="green", opacity=0.15, line_width=0, annotation_text="Fail to Reject Region")
        
    else:  # right-tailed
        crit_val = t.ppf(1 - alpha, df_val)
        fig.add_vrect(x0=crit_val, x1=5, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Rejection Region")
        fig.add_vrect(x0=-5, x1=crit_val, fillcolor="green", opacity=0.15, line_width=0, annotation_text="Fail to Reject Region")

    # Plot the test statistic line
    fig.add_vline(
        x=test_stat, line_dash="dash", line_color=stat_line_color, line_width=3,
        annotation_text=f"Test Statistic: {test_stat:.3f}<br>p-value: {p_value:.4f}",
        annotation_position="top right" if test_stat < 0 else "top left",
        annotation_font_color=stat_line_color
    )

    significance_text = "Statistically Significant" if is_significant else "Not Statistically Significant"
    fig.update_layout(
        title=f"<b>{title}</b><br><sup>Result: {significance_text}</sup>",
        xaxis_title="T-Statistic",
        yaxis_title="Probability Density",
        showlegend=False,
        template="plotly_white",
        margin=dict(t=100)
    )
    return fig

def plot_group_comparison_violin(df, dependent_var, independent_var, title):
    """Creates a colorful and informative plotly violin plot for comparing groups."""
    fig = px.violin(
        df, x=independent_var, y=dependent_var, 
        title=f'<b>{title}</b>', 
        points="all", 
        color=independent_var,
        box=True,
        hover_data=df.columns,
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title=independent_var,
        yaxis_title=dependent_var,
        showlegend=False
    )
    return fig

def plot_correlation_scatter(df, x_var, y_var, title):
    """Creates an enhanced plotly scatter plot with marginal distributions."""
    fig = px.scatter(
        df, x=x_var, y=y_var, 
        title=f'<b>{title}</b>', 
        trendline="ols",
        marginal_x="histogram", 
        marginal_y="histogram",
        hover_data=df.columns,
        color_discrete_sequence=['#1f77b4']  # Set a specific color for points
    )
    
    # Apply marker styling only to the scatter trace
    fig.update_traces(
        selector=dict(type='scatter'),
        marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))
    )
    
    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_var,
        yaxis_title=y_var
    )
    return fig


def plot_contingency_heatmap(df, col1, col2, title):
    """Creates a colorful heatmap for visualizing categorical relationships."""
    contingency_table = pd.crosstab(df[col1], df[col2])
    fig = px.imshow(
        contingency_table,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Viridis',
        title=f"<b>{title}</b>"
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title=col2,
        yaxis_title=col1
    )
    return fig

def plot_distribution_histogram(df, column, title):
    """Creates an enhanced plotly histogram with a mean line and rug plot."""
    mean_val = df[column].mean()
    fig = px.histogram(
        df, x=column, title=f"<b>{title}</b>",
        marginal="rug",
        color_discrete_sequence=['#4c72b0']
    )
    fig.add_vline(
        x=mean_val, line_dash="dash", line_color="red", line_width=2,
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right"
    )
    fig.update_traces(opacity=0.8)
    fig.update_layout(
        template="plotly_white",
        xaxis_title=column,
        yaxis_title="Count"
    )
    return fig

def plot_ordinal_vs_numeric_boxplot(df, numeric_var, ordinal_var, mapping, title):
    """
    Creates a box plot for a numeric variable against an ordinal variable,
    ensuring the ordinal axis is sorted correctly.
    """
    # Sort the category labels by their numeric map value to ensure correct plot order
    # For example, mapping={'EN': 1, 'MI': 2, 'SE': 3} -> sorted_categories=['EN', 'MI', 'SE']
    sorted_categories = sorted(mapping, key=mapping.get)

    fig = px.box(
        df,
        x=ordinal_var,
        y=numeric_var,
        title=f"<b>{title}</b>",
        color=ordinal_var,
        points="all", # 'all' shows the underlying data points as a strip chart
        # This is the key argument to enforce the correct order on the x-axis
        category_orders={ordinal_var: sorted_categories},
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title=ordinal_var,
        yaxis_title=numeric_var,
        showlegend=False
    )
    return fig

def plot_ztest_significance(test_stat, p_value, tail_type, title, alpha=0.05):
    """
    Creates a visually appealing plot of the standard normal distribution,
    highlighting the rejection (red) and acceptance (green) regions for a z-test.
    """
    x_values = np.linspace(-5, 5, 1000)
    y_values = norm.pdf(x_values, 0, 1)  # standard normal N(0,1)
    
    fig = go.Figure()
    
    # Plot the normal distribution curve with fill
    fig.add_trace(go.Scatter(
        x=x_values, y=y_values, mode='lines',
        fill='tozeroy', line_color='rgba(76, 114, 176, 1)', fillcolor='rgba(76, 114, 176, 0.2)',
        name='Standard Normal'
    ))
    
    # Determine significance and color for the test statistic line
    is_significant = p_value <= alpha
    stat_line_color = "red" if is_significant else "green"

    # Define critical values and shade regions
    if tail_type == "two-sided":
        crit_val_upper = norm.ppf(1 - alpha/2)
        crit_val_lower = -crit_val_upper
        fig.add_vrect(x0=crit_val_upper, x1=5, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Rejection Region", annotation_position="top left")
        fig.add_vrect(x0=-5, x1=crit_val_lower, fillcolor="red", opacity=0.2, line_width=0)
        fig.add_vrect(x0=crit_val_lower, x1=crit_val_upper, fillcolor="green", opacity=0.15, line_width=0, annotation_text="Fail to Reject", annotation_position="top")

    elif tail_type == "less":
        crit_val = norm.ppf(alpha)
        fig.add_vrect(x0=-5, x1=crit_val, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Rejection Region")
        fig.add_vrect(x0=crit_val, x1=5, fillcolor="green", opacity=0.15, line_width=0, annotation_text="Fail to Reject Region")
        
    else:  # right-tailed
        crit_val = norm.ppf(1 - alpha)
        fig.add_vrect(x0=crit_val, x1=5, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Rejection Region")
        fig.add_vrect(x0=-5, x1=crit_val, fillcolor="green", opacity=0.15, line_width=0, annotation_text="Fail to Reject Region")

    # Plot the test statistic line
    fig.add_vline(
        x=test_stat, line_dash="dash", line_color=stat_line_color, line_width=3,
        annotation_text=f"Z: {test_stat:.3f}<br>p-value: {p_value:.4f}",
        annotation_position="top right" if test_stat < 0 else "top left",
        annotation_font_color=stat_line_color
    )

    significance_text = "Statistically Significant" if is_significant else "Not Statistically Significant"
    fig.update_layout(
        title=f"<b>{title}</b><br><sup>Result: {significance_text}</sup>",
        xaxis_title="Z-Statistic",
        yaxis_title="Probability Density",
        showlegend=False,
        template="plotly_white",
        margin=dict(t=100)
    )
    return fig

def plot_chi_square_gof(observed, expected, chi2_stat, p_value, title="Chi-Square Goodness-of-Fit"):
    categories = [f"Cat {i+1}" for i in range(len(observed))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories, y=observed, name="Observed",
        marker=dict(color="steelblue")
    ))
    fig.add_trace(go.Bar(
        x=categories, y=expected, name="Expected",
        marker=dict(color="orange")
    ))

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Categories",
        yaxis_title="Counts",
        legend=dict(title="Legend"),
        template="plotly_white",
        annotations=[
            dict(
                x=0.5, y=max(max(observed), max(expected)) * 1.1,
                text=f"Chi2 = {chi2_stat:.3f}, p = {p_value:.4f}",
                showarrow=False, font=dict(size=12)
            )
        ]
    )
    return fig

def plot_wilcoxon_significance(test_stat, p_value, tail_type, title, alpha=0.05):
    """
    Creates a plot for Wilcoxon Signed-Rank Test significance using a normal approximation.
    Highlights the rejection and fail-to-reject regions.
    """
    # Approximate using standard normal for visualization
    x_values = np.linspace(-5, 5, 1000)
    y_values = stats.norm.pdf(x_values)

    fig = go.Figure()

    # Main normal curve
    fig.add_trace(go.Scatter(
        x=x_values, y=y_values, mode='lines',
        fill='tozeroy', line_color='rgba(76, 114, 176, 1)', fillcolor='rgba(76, 114, 176, 0.2)',
        name='Normal Approximation'
    ))

    # Determine significance and line color
    is_significant = p_value <= alpha
    stat_line_color = "red" if is_significant else "green"

    # Plot rejection regions based on tail
    if tail_type == "two-sided":
        crit_val_upper = stats.norm.ppf(1 - alpha/2)
        crit_val_lower = -crit_val_upper
        fig.add_vrect(x0=crit_val_upper, x1=5, fillcolor="red", opacity=0.2, line_width=0)
        fig.add_vrect(x0=-5, x1=crit_val_lower, fillcolor="red", opacity=0.2, line_width=0)
        fig.add_vrect(x0=crit_val_lower, x1=crit_val_upper, fillcolor="green", opacity=0.15, line_width=0)
    elif tail_type == "less":
        crit_val = stats.norm.ppf(alpha)
        fig.add_vrect(x0=-5, x1=crit_val, fillcolor="red", opacity=0.2, line_width=0)
        fig.add_vrect(x0=crit_val, x1=5, fillcolor="green", opacity=0.15, line_width=0)
    else:  # greater
        crit_val = stats.norm.ppf(1 - alpha)
        fig.add_vrect(x0=crit_val, x1=5, fillcolor="red", opacity=0.2, line_width=0)
        fig.add_vrect(x0=-5, x1=crit_val, fillcolor="green", opacity=0.15, line_width=0)

    # Test statistic line
    fig.add_vline(
        x=test_stat, line_dash="dash", line_color=stat_line_color, line_width=3,
        annotation_text=f"Test Statistic: {test_stat:.3f}<br>p-value: {p_value:.4f}",
        annotation_position="top right" if test_stat < 0 else "top left",
        annotation_font_color=stat_line_color
    )

    significance_text = "Statistically Significant" if is_significant else "Not Statistically Significant"
    fig.update_layout(
        title=f"<b>{title}</b><br><sup>Result: {significance_text}</sup>",
        xaxis_title="Wilcoxon Test Statistic (Normal Approximation)",
        yaxis_title="Probability Density",
        showlegend=False,
        template="plotly_white",
        margin=dict(t=100)
    )

    return fig



# --------------------------------------------------------------------------------------------------------------
# Main content area
# --------------------------------------------------------------------------------------------------------------

# Welcome screen view
if not st.session_state.ran_hypothesis:
    st.markdown(
    """
    ## 👋 Welcome to Hypothesis Tester AI

    Upload a dataset, ask a question, and let AI run the right hypothesis test for you.

    ### How it works:
    - 📂 **Upload a CSV file** from the sidebar.
    - 📝 **Ask your question** in the text box below. For the best results, include the relevant column names in parentheses, like this:
        **Example:** *Is the average salary (salary_usd) different for remote employees (remote_ratio)?*
    - 🚀 Click **Run Hypothesis Test**.
    - 📊 Results and plots will appear here.
    - 💬 After the results, you can chat with the AI about your data.
    """
    )

# Results view
else:
    st.subheader("📊 Hypothesis Test Results")
    df = st.session_state.df
    results_response = st.session_state.results['response']
    test_name = results_response.get('test_name')
    params = results_response.get('test_parameters', {})
    test_results = st.session_state.results['test_results']

    plot_context = {}
    
    st.write("#### Data Visualization")
    
    # Plotting logic based on test type
    if test_name in ["Two-Sample Independent t-test", "One-Sample t-test", "Paired t-test"]:
        t_statistic = test_results.get('t_statistic')
        p_value = test_results.get('p_value')
        
        tail = ''

        if params.get('tail', 'not applicable') == 'two-tailed':
            tail = 'two-sided'
        elif params.get('tail', 'not applicable') == 'left-tailed':
            tail = 'less'
        elif params.get('tail', 'not applicable') == 'right-tailed':
            tail = 'greater'

        df_val = test_results.get('degrees_of_freedom')
        
        if all([t_statistic is not None, p_value is not None, tail, df_val is not None]):
            fig = plot_ttest_significance(
                test_stat=t_statistic, p_value=p_value, df_val=df_val,
                tail_type=tail, title=f'Significance of {test_name}'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # CAPTURE PLOT CONTEXT for t-test
            is_significant = p_value <= 0.05
            plot_context = {
                "plot_type": "T-distribution Significance Plot",
                "description": "This plot shows the probability density of the t-distribution for this test. It visualizes whether the result is statistically significant.",
                "variables": {"x_axis": "T-Statistic Value", "y_axis": "Probability Density"},
                "key_observations": {
                    "test_statistic_value": round(t_statistic, 3),
                    "p_value": round(p_value, 4),
                    "degrees_of_freedom": int(df_val),
                    "conclusion": "The test statistic falls into the '{}' region.".format("rejection (red)" if is_significant else "fail-to-reject (green)"),
                    "is_significant": is_significant
                }
            }
        else:
            st.warning("Could not generate t-test significance plot due to missing parameters.")
    
    elif test_name in ["ANOVA", "Kruskal-Wallis H Test", "Mann-Whitney U Test"]:
        dependent_var = params.get('dependent_variable')
        independent_var = params.get('independent_variables', [None])[0]
        if dependent_var and independent_var:
            fig = plot_group_comparison_violin(
                df, dependent_var, independent_var, 
                f"Distribution of {dependent_var} by {independent_var}"
            )
            st.plotly_chart(fig, use_container_width=True)

            # CAPTURE PLOT CONTEXT for ANOVA/group comparison
            group_stats = df.groupby(independent_var)[dependent_var].agg(['mean', 'median', 'std']).round(2).to_dict('index')
            plot_context = {
                "plot_type": "Violin Plot",
                "description": "This plot displays the distribution of a numeric variable across different categories. The shape of the violin shows the density of data points.",
                "variables": {"x_axis": f"Categories of '{independent_var}'", "y_axis": f"Values of '{dependent_var}'"},
                "key_observations": {
                    "groups": list(group_stats.keys()),
                    "group_summary_statistics": group_stats,
                    "visual_trend": "Observe the differences in the median (white dot) and spread (violin shape) for each group."
                }
            }
        else:
            st.warning("Could not generate plot due to missing parameters.")

    elif test_name in ["Correlation Test (Pearson/Spearman)", "Linear Regression Analysis"]:
        ordinal_mapping = params.get("ordinal_mapping")
        cols = results_response.get('columns', [])

        if not cols or len(cols) < 2:
            st.warning("Could not generate plot due to missing column parameters.")
        
        elif ordinal_mapping:
            # Ordinal vs. Numeric case
            ordinal_var = list(ordinal_mapping.keys())[0]
            numeric_var = [col for col in cols if col != ordinal_var][0]
            mapping_dict = ordinal_mapping[ordinal_var]
            fig = plot_ordinal_vs_numeric_boxplot(
                df, numeric_var=numeric_var, ordinal_var=ordinal_var,
                mapping=mapping_dict, title=f"Distribution of {numeric_var} across {ordinal_var} Levels"
            )
            st.plotly_chart(fig, use_container_width=True)

            # CAPTURE PLOT CONTEXT for Ordinal vs. Numeric Box Plot
            group_medians = df.groupby(ordinal_var)[numeric_var].median().round(2).to_dict()
            sorted_categories = sorted(mapping_dict, key=mapping_dict.get)
            plot_context = {
                "plot_type": "Ordered Box Plot",
                "description": "This plot shows the distribution of a numeric variable for each category of an ordered variable. It's used to spot trends.",
                "variables": {"x_axis": f"Ordered categories of '{ordinal_var}'", "y_axis": f"Values of '{numeric_var}'"},
                "key_observations": {
                    "ordered_categories": sorted_categories,
                    "median_by_category": {cat: group_medians.get(cat) for cat in sorted_categories},
                    "correlation_coefficient": round(test_results.get('correlation_coefficient', 0), 3),
                    "visual_trend": "Observe if the median line inside the boxes trends upwards or downwards as you move across the ordered categories from left to right."
                }
            }

        else:
            # Numeric vs. Numeric case
            x_var, y_var = cols[0], cols[1]
            if pd.api.types.is_numeric_dtype(df[x_var]) and pd.api.types.is_numeric_dtype(df[y_var]):
                fig = plot_correlation_scatter(df, x_var, y_var, f"Relationship between {x_var} and {y_var}")
                st.plotly_chart(fig, use_container_width=True)

                # CAPTURE PLOT CONTEXT for Scatter Plot
                plot_context = {
                    "plot_type": "Scatter Plot with Trendline",
                    "description": "This plot shows the relationship between two numeric variables. Each dot is a data point.",
                    "variables": {"x_axis": x_var, "y_axis": y_var},
                    "key_observations": {
                        "correlation_coefficient": round(test_results.get('correlation_coefficient', 0), 3),
                        "p_value": round(test_results.get('p_value', 1), 4),
                        "visual_trend": "Observe the direction of the points and the slope of the central trendline. An upward slope suggests a positive correlation, while a downward slope suggests a negative one."
                    }
                }
            else:
                st.warning(f"Could not generate scatter plot. Both '{x_var}' and '{y_var}' must be numeric.")

    elif test_name in ["Chi-Square Test of Independence", "Fisher's Exact Test"]:
        vars_to_plot = params.get('independent_variables', [])
        if len(vars_to_plot) >= 2:
            col1, col2 = vars_to_plot[0], vars_to_plot[1]
            fig = plot_contingency_heatmap(df, col1, col2, f"Contingency Heatmap of {col1} and {col2}")
            st.plotly_chart(fig, use_container_width=True)

            # CAPTURE PLOT CONTEXT for Heatmap
            contingency_table = pd.crosstab(df[col1], df[col2])
            plot_context = {
                "plot_type": "Contingency Heatmap",
                "description": "This plot visualizes the frequency of co-occurrence between two categorical variables. The color of each cell indicates the count.",
                "variables": {"x_axis": col2, "y_axis": col1},
                "key_observations": {
                    "contingency_table": contingency_table.to_dict(),
                    "visual_trend": "Look for cells or rows/columns that are significantly darker or lighter than others. This suggests a pattern or association between the variables."
                }
            }
        else:
            st.warning("Could not generate plot. Chi-Square test requires at least two categorical variables.")
    
    elif test_name in ["One-Sample Z-test"]:
        z_statistic = test_results.get('z_statistic')
        p_value = test_results.get('p_value')
        
        tail = ''
        if params.get('tail', 'not applicable') == 'two-tailed':
            tail = 'two-sided'
        elif params.get('tail', 'not applicable') == 'left-tailed':
            tail = 'less'
        elif params.get('tail', 'not applicable') == 'right-tailed':
            tail = 'greater'

        if all([z_statistic is not None, p_value is not None, tail]):
            fig = plot_ztest_significance(
                test_stat=z_statistic, p_value=p_value, tail_type=tail,
                title=f'Significance of {test_name}'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # CAPTURE PLOT CONTEXT for z-test
            is_significant = p_value <= 0.05
            plot_context = {
                "plot_type": "Z-distribution Significance Plot",
                "description": "This plot shows the probability density of the standard normal distribution for this test. It visualizes whether the result is statistically significant.",
                "variables": {"x_axis": "Z-Statistic Value", "y_axis": "Probability Density"},
                "key_observations": {
                    "test_statistic_value": round(z_statistic, 3),
                    "p_value": round(p_value, 4),
                    "conclusion": "The test statistic falls into the '{}' region.".format("rejection (red)" if is_significant else "fail-to-reject (green)"),
                    "is_significant": is_significant
                }
            }
        else:
            st.warning("Could not generate z-test significance plot due to missing parameters.")

    elif test_name in ["Chi-Square Goodness-of-Fit Test"]:
        chi2_stat = test_results.get('chi2_statistic')
        p_value = test_results.get('p_value')
        expected_values = params.get('expected_values', [])
        columns = results_response.get('columns', [])
        
        # we assume observed values come from df counts on categorical col
        if columns and expected_values:
            col_name = columns[0]
            observed_counts = df[col_name].value_counts().sort_index().tolist()

            if all([chi2_stat is not None, p_value is not None, observed_counts, expected_values]):
                fig = plot_chi_square_gof(
                    observed=observed_counts,
                    expected=expected_values,
                    chi2_stat=chi2_stat,
                    p_value=p_value,
                    title=f"{test_name}: Observed vs Expected"
                )
                st.plotly_chart(fig, use_container_width=True)

                # CAPTURE PLOT CONTEXT
                is_significant = p_value <= 0.05
                plot_context = {
                    "plot_type": "Chi-Square Goodness-of-Fit Plot",
                    "description": "This plot compares observed and expected category frequencies to evaluate how well the data fits the expected distribution.",
                    "variables": {"x_axis": "Categories", "y_axis": "Counts"},
                    "key_observations": {
                        "chi2_statistic_value": round(chi2_stat, 3),
                        "p_value": round(p_value, 4),
                        "conclusion": "The distribution {} the expected one.".format(
                            "differs significantly from" if is_significant else "does not differ significantly from"
                        ),
                        "is_significant": is_significant
                    }
                }
            else:
                st.warning("Could not generate Chi-Square GOF plot due to missing parameters.")

    elif test_name in ["Wilcoxon Signed-Rank Test"]:
        w_statistic = test_results.get('t_statistic')
        p_value = test_results.get('p_value')
        
        tail = ''
        if params.get('tail', 'not applicable') == 'two-tailed':
            tail = 'two-sided'
        elif params.get('tail', 'not applicable') == 'left-tailed':
            tail = 'less'
        elif params.get('tail', 'not applicable') == 'right-tailed':
            tail = 'greater'

        if all([w_statistic is not None, p_value is not None]):
            fig = plot_wilcoxon_significance(
                test_stat=w_statistic,
                p_value=p_value,
                tail_type=tail,
                title=f'Significance of {test_name}'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # CAPTURE PLOT CONTEXT for Wilcoxon Signed-Rank Test
            is_significant = p_value <= 0.05
            plot_context = {
                "plot_type": "Wilcoxon Signed-Rank Test Plot",
                "description": "This plot shows the distribution of paired differences and highlights the Wilcoxon test statistic to indicate significance.",
                "variables": {"x_axis": "Difference Values", "y_axis": "Frequency"},
                "key_observations": {
                    "test_statistic_value": round(w_statistic, 3),
                    "p_value": round(p_value, 4),
                    "conclusion": "The test statistic falls into the '{}' region.".format(
                        "rejection (red)" if is_significant else "fail-to-reject (green)"
                    ),
                    "is_significant": is_significant
                }
            }
        else:
            st.warning("Could not generate Wilcoxon Signed-Rank significance plot due to missing parameters.")



    # LLM Summary once the graph is plotted 
    st.write("#### Summary:")
    
    user_hypothesis_question = st.session_state.hypotheses_quest
    llm_response = st.session_state.results['response']
    test_results = st.session_state.results['test_results']

    summ_placeholder = st.empty()

    res = ''
    
    for chunks in summarize(user_hypothesis_question, llm_response, test_results, plot_context):
        res += chunks
        summ_placeholder.markdown(res + "▌")
    summ_placeholder.markdown(res)    

    st.session_state.plotContext = plot_context
