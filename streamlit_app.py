import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz

# -------------------------
# Helper: fuzzy match for smart mode
# -------------------------
def fuzzy_match(query, choices, threshold=70):
    if not choices:
        return None
    match, score, _ = process.extractOne(query, choices)
    return match if score >= threshold else None


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Hypothesis Tester AI", layout="wide")

st.title("🤖 Hypothesis Tester AI")

# Step 1: Upload file
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Preview:", df.head())

    # Separate column types
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    # Step 2: Guided Mode (explicit column selection)
    st.subheader("🛠 Guided Mode: Select Columns")
    selected_cols = st.multiselect("Choose columns for hypothesis testing", df.columns.tolist())

    # Step 3: User enters natural language question
    st.subheader("💬 Ask Your Question")
    user_prompt = st.text_area(
        "Enter your hypothesis in plain English (e.g., 'Was sales higher last year than this year?')"
    )
    

    # Step 4: Run
    if st.button("🔎 Run Hypothesis Test"):
        if not user_prompt:
            st.warning("Please enter a question before running!")
        else:
            # Case 1: User selected columns (Guided Mode)
            if selected_cols:
                st.success("✅ Guided Mode Activated")
                st.info(f"📂 Selected columns: {selected_cols}")
                st.write(f"➡️ Interpreting question: {user_prompt}")
                # TODO: Call LLM to decide test type + tails
                # TODO: Pass to hypothesis_test.py
                st.write("📊 Running statistical test with chosen columns...")

            # Case 2: No columns selected → Smart Mode
            else:
                st.success("🤖 Smart Mode Activated (auto-detecting columns)")
                st.write(f"➡️ Interpreting question: {user_prompt}")

                # Try to fuzzy match user words with dataset columns
                matched_cols = [
                    col for col in df.columns
                    if fuzz.partial_ratio(col.lower(), user_prompt.lower()) >= 70
                ]

                if matched_cols:
                    st.info(f"📂 Auto-selected columns: {matched_cols}")
                    # TODO: LLM decides test type + tails
                    # TODO: Pass matched_cols to hypothesis_test.py
                    st.write("📊 Running statistical test with auto-matched columns...")
                else:
                    st.error("⚠️ Could not infer columns from your question. Please try Guided Mode.")

