import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
import re
from modules.agent_layer import handle

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

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("📊 Preview:", df.head())

            # Separate column types
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

            # Guided Mode
            st.subheader("🛠 Guided Mode: Select Columns")
            selected_cols = st.multiselect(
                "Choose columns for hypothesis testing (Skip if you don't know)",
                df.columns.tolist()
            )

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
                    
                    st.info('Selecting the correct Hypothesis Test...')
                    
                    # moving forward to LLM layer to pass these values to work with and get the suggested test
                    response = handle(user_prompt, df, matched_cols)

                    # setting a dummy result as of now to show on the plot space
                    st.session_state.results = {"cols": matched_cols, "prompt": user_prompt, 'response' : response}
                    st.session_state.ran_hypothesis = True

                # Fallback to fuzzy search if no parentheses were used
                else:
                    # extracting the column names using fuzzy search
                    matched_cols = [
                        col for col in df.columns
                        if fuzz.partial_ratio(col.lower(), user_prompt.lower()) >= 70
                    ]
                    if matched_cols:
                        st.success("Inferred Columns :", matched_cols)
                        
                        st.info('Selecting the correct Hypothesis Test...')

                        # passing the user prompt, and dataframe to LLM to identify correct Test for this 
                        response = handle(user_prompt, df, matched_cols)    
                        
                        st.session_state.results = {"cols": matched_cols, "prompt": user_prompt, 'respnonse' : response}

                    else:
                        st.error("⚠️ Could not infer columns. Try putting the relevant columns into parenthesis(column_name).")
                        st.stop()

                    st.session_state.ran_hypothesis = True

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


# -------------------------
# Main content area
# -------------------------
if not st.session_state.ran_hypothesis:
    # Show welcome screen
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

else:
    # Results view
    st.subheader("📊 Hypothesis Test Results")

    tab1, tab2 = st.tabs(["📈 Plots", "📜 Test Summary"])
    with tab1:
        st.info("This is where plots will appear.")
        # TODO: Replace with st.plotly_chart(fig) later
    with tab2:
        st.write("### Summary")
        st.json(st.session_state.results)  # placeholder for now
