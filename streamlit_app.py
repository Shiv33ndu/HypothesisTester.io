import streamlit as st
import pandas as pd

st.title("🤖 Hypothesis Tester AI")

# Step 1: Upload file
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Preview:", df.head())

    # Step 2: Select columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    selected_cols = st.multiselect("🔍 Select Columns for Hypothesis Testing", df.columns.tolist())

    # Step 3: User enters natural language question
    user_prompt = st.text_area(
        "💬 Ask a question about your data (e.g. 'Is last year better than this year?')"
    )

    # Step 4: Submit
    if st.button("Run Hypothesis Test"):
        if not selected_cols or not user_prompt:
            st.warning("Please select columns and enter a question first!")
        else:
            # TODO: Pass `selected_cols` + `user_prompt` → LLM interpreter → hypothesis_test.py
            st.success(f"🔎 Your question: {user_prompt}")
            st.info(f"📂 Selected columns: {selected_cols}")
            # Example placeholder
            st.write("➡️ Running hypothesis test (LLM + Stats)...")
