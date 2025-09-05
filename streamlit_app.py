import streamlit as st
import pandas as pd
from rapidfuzz import fuzz

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

# -------------------------
# Sidebar UI
# -------------------------
# with st.sidebar:
#     st.title("⚙️ Controls")

#     # Step 1: Upload file
#     uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         st.write("📊 Preview:", df.head())

#         # Separate column types
#         numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
#         categorical_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

#         # Step 2: Guided Mode (explicit column selection)
#         st.subheader("🛠 Guided Mode: Select Columns")
#         selected_cols = st.multiselect(
#             "Choose columns for hypothesis testing (Skip if you don't know)",
#             df.columns.tolist()
#         )

#         # Step 3: User enters natural language question
#         st.subheader("💬 Ask Your Question")
#         user_prompt = st.text_area(
#             "Enter your hypothesis in plain English (e.g., 'Was sales higher last year than this year?')"
#         )

#         # Step 4: Run
#         if st.button("🔎 Run Hypothesis Test"):
#             if not user_prompt:
#                 st.warning("Please enter a question before running!")
#             else:
#                 if selected_cols:
#                     st.success("✅ Guided Mode Activated")
#                     st.info(f"📂 Selected columns: {selected_cols}")
#                     st.write(f"➡️ Interpreting question: {user_prompt}")
#                     # TODO: Call LLM + hypothesis_test.py
#                     st.session_state.results = {"mode": "guided", "cols": selected_cols, "prompt": user_prompt}

#                 else:
#                     st.success("🤖 Smart Mode Activated (auto-detecting columns)")
#                     st.write(f"➡️ Interpreting question: {user_prompt}")

#                     matched_cols = [
#                         col for col in df.columns
#                         if fuzz.partial_ratio(col.lower(), user_prompt.lower()) >= 70
#                     ]

#                     if matched_cols:
#                         st.info(f"📂 Auto-selected columns: {matched_cols}")
#                         # TODO: Call LLM + hypothesis_test.py
#                         st.session_state.results = {"mode": "smart", "cols": matched_cols, "prompt": user_prompt}
#                     else:
#                         st.error("⚠️ Could not infer columns from your question. Please try Guided Mode.")

#                 st.session_state.ran_hypothesis = True

#     # Step 5: Chat box appears only after running hypothesis
#     if st.session_state.ran_hypothesis:
#         st.subheader("💬 Chat with Hypothesis AI")
#         chat_prompt = st.text_area("Ask follow-up questions about results")
#         if st.button("Send"):
#             # TODO: Call LLM with st.session_state.results + chat_prompt
#             st.write("🤖 AI: This is where the response will appear.")

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
                "Enter your hypothesis in plain English (e.g., 'Was sales higher last year than this year?')"
            )

            # Run Hypothesis
            if st.button("🔎 Run Hypothesis Test"):
                if not user_prompt:
                    st.warning("Please enter a question before running!")
                else:
                    handle(user_prompt, df)

                    if selected_cols:
                        st.success("✅ Guided Mode Activated")
                        st.session_state.results = {"mode": "guided", "cols": selected_cols, "prompt": user_prompt}
                    else:
                        matched_cols = [
                            col for col in df.columns
                            if fuzz.partial_ratio(col.lower(), user_prompt.lower()) >= 70
                        ]
                        if matched_cols:
                            st.success("🤖 Smart Mode Activated (auto-detecting columns)")
                            st.session_state.results = {"mode": "smart", "cols": matched_cols, "prompt": user_prompt}
                        else:
                            st.error("⚠️ Could not infer columns. Try Guided Mode.")
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
        - 📂 Upload a CSV file from the sidebar  
        - 🛠 Either choose columns yourself (Guided Mode)  
        - 🤖 Or just ask in plain English (Smart Mode)  
        - 🔎 Click **Run Hypothesis Test**  
        - 📊 Results and plots will appear here  
        - 💬 After results, chat with the AI about your data
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
