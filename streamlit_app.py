# import streamlit as st
# import pandas as pd
# import numpy as np
# from rapidfuzz import fuzz
# import re
# from modules.agent_layer import handle
# import plotly.express as px
# import plotly.graph_objects as go
# from scipy.stats import t, norm

# from modules.agent_layer import check_user_question

# # -------------------------
# # Streamlit UI
# # -------------------------
# st.set_page_config(page_title="Hypothesis Tester AI", layout="wide")

# st.title("🤖 Hypothesis Tester AI")

# # -------------------------
# # Custom sidebar width
# # -------------------------
# st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"] {
#         width: 30% !important;
#     }
#     [data-testid="stSidebarContent"] {
#         width: 100% !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # -------------------------
# # Track app state
# # -------------------------
# if "ran_hypothesis" not in st.session_state:
#     st.session_state.ran_hypothesis = False
# if "results" not in st.session_state:
#     st.session_state.results = None

# # ------------------------------------
# # Column extraction from parenthesis
# # ------------------------------------
# def extract_from_parentheses(text):
#     """
#     Extracts all text enclosed in parentheses from a string.
#     Returns a list of the found strings.
#     """
#     return re.findall(r'\((.*?)\)', text)




# # -------------------------
# # Sidebar UI
# # -------------------------

# with st.sidebar:
#     st.title("⚙️ Controls")

#     # Tabs inside sidebar
#     tab1, tab2 = st.tabs(["🛠 Hypothesis Setup", "💬 Chat"])

#     # -------------------------
#     # TAB 1: Setup & Run
#     # -------------------------
#     with tab1:
#         uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

#         if uploaded_file is None:
#             st.session_state.ran_hypothesis = False

#         if uploaded_file is not None:
#             df = pd.read_csv(uploaded_file)
#             st.write("📊 Preview:", df.head())

#             # Separate column types
#             numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
#             categorical_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

#             # User question
#             st.subheader("💬 Ask Your Question")
#             user_prompt = st.text_area(
#                 "Enter your hypothesis in plain English,. Hint: using column names in parenthesis() will give better results",

#                 placeholder="Is benefit score(benefits_score) negatively correlated with salary(salary_usd)?"
#                 )

#             # Run Hypothesis
#             if st.button("🔎 Run Hypothesis Test"):
#                 if not user_prompt:
#                     st.warning("Please enter a question before running!")
                
#                 # New logic to handle user-enforced format
#                 enclosed_cols = extract_from_parentheses(user_prompt)
                
#                 # Check if the user used the recommended parenthesis format
#                 if enclosed_cols:
#                     # User provided column names, let's validate them
#                     matched_cols = [col for col in enclosed_cols if col in df.columns]
                    
#                     # Check for typos
#                     if len(matched_cols) != len(enclosed_cols):
#                         st.error("⚠️ Column name mismatch! One or more of the columns you entered do not exist in the dataset. Please check for typos and try again.")

#                         # 1. Identify the columns with typos
#                         typo_cols = [col for col in enclosed_cols if col not in df.columns]
                        
#                         # Generate the narrowed, fuzzy-matched hints
#                         fuzzy_matches = []
#                         for user_col in typo_cols:
#                             # Find the best fuzzy match for this specific column name
#                             best_match = None
#                             highest_ratio = 0
                            
#                             for df_col in df.columns:
#                                 ratio = fuzz.ratio(user_col.lower(), df_col.lower())
#                                 if ratio > highest_ratio:
#                                     highest_ratio = ratio
#                                     best_match = df_col
                            
#                             # Add the best match if it's above a certain threshold (e.g., 70)
#                             if highest_ratio >= 70 and best_match not in fuzzy_matches:
#                                 fuzzy_matches.append(f"{user_col} -> {best_match}")

#                         st.error(f"Wrong column name you entered : {typo_cols} ")

#                         if fuzzy_matches:
#                             st.info(f"Did you mean: {', '.join(fuzzy_matches)} ?")

#                         st.stop()
                    
#                     # If all columns match, proceed with the validated list
#                     st.success(f"Columns found from your input : {matched_cols}")
#                     info_placeholder = st.empty()
#                     info_placeholder.info('🔍  Choosing the right test …')
                    
#                     # moving forward to LLM layer to pass these values to work with and get the suggested test
#                     check_question = check_user_question(user_prompt, df) 
                    
#                     info_placeholder.empty()          # remove old banner

#                     if check_question["decision"] == "descriptive":
#                         with info_placeholder.container():
#                             st.warning("⚠️  Not a hypothesis-testing question.")
#                             st.markdown(
#                                 f"*Descriptive answer:*  \n"
#                                 f"**{check_question['answer']}**"
#                             )
#                             with st.expander("Why no test was run"):
#                                 st.write(check_question["reason"])
                    
#                     else:
#                         response, test_results = handle(user_prompt, df, matched_cols)   
                    
#                         # print('in code')
#                         # print(response)  

#                         # setting a dummy result as of now to show on the plot space
#                         st.session_state.results = {"cols": matched_cols, "prompt": user_prompt, 'response' : response, 'test_results' : test_results}
#                         st.session_state.ran_hypothesis = True

#                 # Fallback to fuzzy search if no parentheses were used
#                 else:
#                     # extracting the column names using fuzzy search
#                     matched_cols = [
#                         col for col in df.columns
#                         if fuzz.partial_ratio(col.lower(), user_prompt.lower()) >= 70
#                     ]
#                     if matched_cols:
#                         st.success(f"Inferred Columns : {matched_cols}")
                        
#                         info_placeholder = st.empty()

#                         info_placeholder.info('🔍  Choosing the right test …')

#                         # passing the user prompt, and dataframe to LLM to identify correct Test for this 
#                         check_question = check_user_question(user_prompt, df) 
                    
#                         info_placeholder.empty()          # remove old banner

#                         if check_question["decision"] == "descriptive":
#                             with info_placeholder.container():
#                                 st.warning("⚠️  Not a hypothesis-testing question.")
#                                 st.markdown(
#                                     f"*Descriptive answer:*  \n"
#                                     f"**{check_question['answer']}**"
#                                 )
#                                 with st.expander("Why no test was run"):
#                                     st.write(check_question["reason"])

#                         else:
#                             response, test_results = handle(user_prompt, df, matched_cols)   
                        
#                             # print('in code')
#                             # print(response)  

#                             # setting a dummy result as of now to show on the plot space
#                             st.session_state.results = {"cols": matched_cols, "prompt": user_prompt, 'response' : response, 'test_results' : test_results}
                        
#                             st.session_state.ran_hypothesis = True

#                     else:
#                         st.error("⚠️ Could not infer columns. Try putting the relevant columns into parenthesis(column_name).")
#                         st.stop()

                    

#     # -------------------------
#     # TAB 2: Chat (only after hypothesis ran)
#     # -------------------------
#     with tab2:
#         if st.session_state.get("ran_hypothesis", False):
#             # Init chat history
#             if "messages" not in st.session_state:
#                 st.session_state.messages = []

#             st.subheader("💬 Chat with Hypothesis AI")

#             # Display chat history
#             for msg in st.session_state.messages:
#                 if msg["role"] == "user":
#                     st.markdown(f"🧑 **You:** {msg['content']}")
#                 else:
#                     st.markdown(f"🤖 **AI:** {msg['content']}")

#             # Chat input
#             user_input = st.text_area("Type your message:", key="chat_input", height=80)
#             if st.button("Send", key="chat_send"):
#                 if user_input.strip():
#                     st.session_state.messages.append({"role": "user", "content": user_input})

#                     # TODO: Call your LLM here
#                     ai_reply = f"(AI placeholder reply to: '{user_input}')"
#                     st.session_state.messages.append({"role": "ai", "content": ai_reply})

#                     st.session_state.chat_input = ""
#                     st.experimental_rerun()
#         else:
#             st.info("⚠️ Run a hypothesis test first to enable chat.")




# # --------------------------------------------------------------------------------------------------------------
# # Main content area
# # --------------------------------------------------------------------------------------------------------------


# # =========================================================================
# # PLOTTING FUNCTIONS
# # =========================================================================

# def create_grouped_boxplot(df, dependent_var, independent_var, title):
#     """Creates a plotly box plot for comparing groups."""
#     fig = px.box(df, x=independent_var, y=dependent_var, 
#                  title=title, 
#                  points="all", 
#                  color=independent_var)
#     return fig

# def create_scatter_plot(df, x_var, y_var, title):
#     """Creates a plotly scatter plot for correlation/regression."""
#     fig = px.scatter(df, x=x_var, y=y_var, 
#                      title=title, 
#                      trendline="ols",
#                      hover_data=df.columns)
#     return fig

# def create_bar_chart(df, x_var, y_var, title):
#     """Creates a plotly bar chart for visualizing categorical relationships."""
#     fig = px.bar(df, x=x_var, y=y_var, 
#                  title=title, 
#                  text_auto='.2s')
#     return fig

# def create_histogram(df, column, title):
#     """Creates a plotly histogram for a single variable."""
#     fig = px.histogram(df, x=column, title=title)
#     return fig

# def create_significance_plot_for_ttest(dependent_var_data, test_stat, p_value, tail_type, title, alpha=0.05):
#     """
#     Creates a plot visualizing the t-distribution with significance regions.
#     The test statistic line is colored based on the p-value.
#     """
#     df_val = len(dependent_var_data) - 1 # Degrees of freedom
#     x_values = np.linspace(-4, 4, 1000)
#     y_values = t.pdf(x_values, df_val)
    
#     fig = go.Figure()
    
#     # Plot the t-distribution curve
#     fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='t-Distribution'))
    
#     # Calculate critical values
#     if tail_type == "two-tailed":
#         critical_value = t.ppf(1 - alpha/2, df_val)
#         # Rejection region
#         fig.add_vrect(x0=critical_value, x1=4, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Rejection Region", annotation_position="top left")
#         fig.add_vrect(x0=-4, x1=-critical_value, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Rejection Region", annotation_position="top right")
#     elif tail_type == "left-tailed":
#         critical_value = t.ppf(alpha, df_val)
#         fig.add_vrect(x0=-4, x1=critical_value, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Rejection Region", annotation_position="top right")
#     else: # right-tailed
#         critical_value = t.ppf(1 - alpha, df_val)
#         fig.add_vrect(x0=critical_value, x1=4, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Rejection Region", annotation_position="top left")
        
#     # Determine significance and color
#     is_significant = p_value <= alpha
#     stat_line_color = "red" if is_significant else "green"
    
#     # Plot the test statistic line
#     fig.add_vline(x=test_stat, line_dash="dash", line_color=stat_line_color, 
#                   annotation_text=f"Test Stat: {test_stat:.2f} (p={p_value:.3f})",
#                   annotation_position="bottom right")

#     # Add a title based on the result
#     significance_text = "Statistically Significant" if is_significant else "Not Statistically Significant"
#     fig.update_layout(title=f"{title}<br><sup>Result: {significance_text}</sup>",
#                       xaxis_title="T-Statistic",
#                       yaxis_title="Probability Density",
#                       showlegend=False)

#     return fig



# # -----------------------------
# # MAIN CONTENT VIEW TAB CODES
# # -----------------------------

# if not st.session_state.ran_hypothesis:
#     # Show welcome screen
#     st.markdown(
#     """
#     ## 👋 Welcome to Hypothesis Tester AI

#     Upload a dataset, ask a question, and let AI run the right hypothesis test for you.

#     ### How it works:
#     - 📂 **Upload a CSV file** from the sidebar.
#     - 📝 **Ask your question** in the text box below. For the best results, include the relevant column names in parentheses, like this:
#         **Example:** *Is the average salary (salary_usd) different for remote employees (remote_ratio)?*
#     - 🚀 Click **Run Hypothesis Test**.
#     - 📊 Results and plots will appear here.
#     - 💬 After the results, you can chat with the AI about your data.
#     """
#     )

# # Results view
# else:
#     st.subheader("📊 Hypothesis Test Results")
#     results = st.session_state.results['response']
#     print('In streamlit')
#     print(results)
#     test_name = results.get('test_name')
#     params = results.get('test_parameters', {})
#     test_results = st.session_state.results['test_results'].get('test_results', {})

#     tab1, tab2 = st.tabs(["📈 Plots", "📜 Test Summary"])
#     with tab1:
#         st.write("#### Data Visualization")
        
#         # Plotting logic based on test type
#         if test_name in ["Two-Sample Independent t-test", "One-Sample t-test"]:
#             dependent_var = params.get('dependent_variable')
#             independent_var = params.get('independent_variables', [None])[0]
#             t_statistic = test_results.get('t_statistic')
#             p_value = test_results.get('p_value')
#             tail = params.get('tail')
            
#             if all([dependent_var, independent_var, t_statistic is not None, p_value is not None, tail]):
#                 fig = create_significance_plot_for_ttest(df, dependent_var, independent_var, t_statistic, p_value, tail)
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.warning("Could not generate t-test significance plot due to missing parameters.")
        
#         elif test_name in ["ANOVA", "Kruskal-Wallis H Test", "Mann-Whitney U Test"]:
#             dependent_var = params.get('dependent_variable')
#             independent_var = params.get('independent_variables', [None])[0]
#             if dependent_var and independent_var:
#                 fig = create_grouped_boxplot(df, dependent_var, independent_var, 
#                                              f"Distribution of {dependent_var} by {independent_var}")
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.warning("Could not generate plot due to missing parameters.")
        
#         elif test_name in ["Correlation Test (Pearson/Spearman)", "Linear Regression Analysis"]:
#             x_var = results.get('columns')[0]
#             y_var = results.get('columns')[1]
#             if x_var and y_var:
#                 fig = create_scatter_plot(df, x_var, y_var,
#                                           f"Relationship between {x_var} and {y_var}")
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.warning("Could not generate plot due to missing parameters.")

#         elif test_name in ["Chi-Square Test of Independence", "Fisher's Exact Test"]:
#             col1 = results.get('columns')[0]
#             col2 = results.get('columns')[1]
#             if col1 and col2:
#                 contingency_table = pd.crosstab(df[col1], df[col2])
#                 fig = create_bar_chart(contingency_table, col1, col2,
#                                        f"Contingency Table of {col1} and {col2}")
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.warning("Could not generate plot due to missing parameters.")
        
#         else: # Default plot for single-variable tests
#             dependent_var = params.get('dependent_variable')
#             if dependent_var:
#                 fig = create_histogram(df, dependent_var, f"Distribution of {dependent_var}")
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.warning("Could not generate a suitable plot for this test.")

#     with tab2:
#         st.write("#### Summary")
#         # TODO: Replace with LLM-generated summary for better readability
#         st.markdown(f"**Test Name:** `{results.get('test_name')}`")
#         st.markdown(f"**Reasoning:** `{results.get('reasoning')}`")
#         st.write("---")
#         st.write("#### Raw Results (for developers)")
#         st.json(results)

import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import re
from modules.agent_layer import handle
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import t, norm

from modules.agent_layer import check_user_question

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

    # Tabs inside sidebar
    tab1, tab2 = st.tabs(["🛠 Hypothesis Setup", "💬 Chat"])

    # -------------------------
    # TAB 1: Setup & Run
    # -------------------------
    with tab1:
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
                        response, test_results = handle(user_prompt, df, matched_cols)  
                    
                        # setting a dummy result as of now to show on the plot space
                        st.session_state.results = {"cols": matched_cols, "prompt": user_prompt, 'response' : response, 'test_results' : test_results}
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
                            response, test_results = handle(user_prompt, df, matched_cols)  
                        
                            # setting a dummy result as of now to show on the plot space
                            st.session_state.results = {"cols": matched_cols, "prompt": user_prompt, 'response' : response, 'test_results' : test_results}
                        
                            st.session_state.ran_hypothesis = True

                    else:
                        st.error("⚠️ Could not infer columns. Try putting the relevant columns into parenthesis(column_name).")
                        st.stop()
    # -------------------------
    # TAB 2: Chat (only after hypothesis ran)
    # -------------------------
    with tab2:
        if st.session_state.get("ran_hypothesis", False):
            # Init chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            st.subheader("💬 Chat with Hypothesis AI")

            # Display chat history
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"🧑 **You:** {msg['content']}")
                else:
                    st.markdown(f"🤖 **AI:** {msg['content']}")

            # Chat input
            user_input = st.text_area("Type your message:", key="chat_input", height=80)
            if st.button("Send", key="chat_send"):
                if user_input.strip():
                    st.session_state.messages.append({"role": "user", "content": user_input})

                    # TODO: Call your LLM here
                    ai_reply = f"(AI placeholder reply to: '{user_input}')"
                    st.session_state.messages.append({"role": "ai", "content": ai_reply})

                    st.session_state.chat_input = ""
                    st.experimental_rerun()
        else:
            st.info("⚠️ Run a hypothesis test first to enable chat.")


# =========================================================================
# NEW AND IMPROVED PLOTTING FUNCTIONS
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
    if tail_type == "two-tailed":
        crit_val_upper = t.ppf(1 - alpha/2, df_val)
        crit_val_lower = -crit_val_upper
        # Red rejection regions
        fig.add_vrect(x0=crit_val_upper, x1=5, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Rejection Region", annotation_position="top left")
        fig.add_vrect(x0=-5, x1=crit_val_lower, fillcolor="red", opacity=0.2, line_width=0)
        # Green "Fail to Reject" region
        fig.add_vrect(x0=crit_val_lower, x1=crit_val_upper, fillcolor="green", opacity=0.15, line_width=0, annotation_text="Fail to Reject Region", annotation_position="top")

    elif tail_type == "left-tailed":
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
        color_discrete_sequence=['#1f77b4'] # Set a specific color for points
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
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
    test_results = st.session_state.results['test_results'].get('test_results', {})

    tab1, tab2 = st.tabs(["📈 Plots", "📜 Test Summary"])
    with tab1:
        st.write("#### Data Visualization")
        
        # Plotting logic based on test type
        if test_name in ["Two-Sample Independent t-test", "One-Sample t-test", "Paired Sample t-test"]:
            t_statistic = test_results.get('t_statistic')
            p_value = test_results.get('p_value')
            tail = params.get('tail')
            df_val = test_results.get('degrees_of_freedom')
            
            if all([t_statistic is not None, p_value is not None, tail, df_val is not None]):
                fig = plot_ttest_significance(
                    test_stat=t_statistic, 
                    p_value=p_value, 
                    df_val=df_val,
                    tail_type=tail, 
                    title=f'Significance of {test_name}'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate t-test significance plot due to missing parameters (t-statistic, p-value, tail, or degrees of freedom).")
        
        elif test_name in ["ANOVA", "Kruskal-Wallis H Test", "Mann-Whitney U Test"]:
            dependent_var = params.get('dependent_variable')
            independent_var = params.get('independent_variables', [None])[0]
            if dependent_var and independent_var:
                fig = plot_group_comparison_violin(
                    df, dependent_var, independent_var, 
                    f"Distribution of {dependent_var} by {independent_var}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate plot due to missing parameters.")
        
        elif test_name in ["Correlation Test (Pearson/Spearman)", "Linear Regression Analysis"]:
            # Assuming the columns are in 'independent_variables' and 'dependent_variable' from params
            x_var = params.get('independent_variables', [None])[0]
            y_var = params.get('dependent_variable')
            if x_var and y_var:
                fig = plot_correlation_scatter(
                    df, x_var, y_var,
                    f"Relationship between {x_var} and {y_var}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate plot due to missing parameters (independent/dependent variables).")

        elif test_name in ["Chi-Square Test of Independence", "Fisher's Exact Test"]:
            # Chi-square usually involves two categorical variables
            vars_to_plot = params.get('independent_variables', [])
            if len(vars_to_plot) >= 2:
                col1, col2 = vars_to_plot[0], vars_to_plot[1]
                fig = plot_contingency_heatmap(
                    df, col1, col2,
                    f"Contingency Heatmap of {col1} and {col2}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate plot. Chi-Square test requires at least two categorical variables.")
        
        else: # Default plot for single-variable tests like Shapiro-Wilk
            dependent_var = params.get('dependent_variable')
            if dependent_var:
                fig = plot_distribution_histogram(df, dependent_var, f"Distribution of {dependent_var}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate a suitable plot for this test.")

    with tab2:
        st.write("#### Summary")
        # TODO: Replace with LLM-generated summary for better readability
        st.markdown(f"**Test Name:** `{results_response.get('test_name')}`")
        st.markdown(f"**Reasoning:** `{results_response.get('reasoning')}`")
        st.write("---")
        st.write("#### Raw Test Results")
        st.json(test_results)
        st.write("#### Raw LLM Response")
        st.json(results_response)