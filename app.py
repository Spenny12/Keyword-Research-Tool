import streamlit as st
import pandas as pd
from google import genai
from google.genai import types

# --- UI Configuration ---
st.set_page_config(page_title="SEO KW Classifier", layout="wide")
st.title("KW Classifier")
st.write("Upload a CSV to label keyword intent")

# --- Sidebar ---
with st.sidebar:
    gemini_api_key = st.text_input("Enter Gemini API Key", type="password")
    st.divider()
    st.info("**Model:** Gemini 3 Flash Preview")
    st.info("**Temperature:** 0.0 (Locked)")
    st.info("**Thinking:** Minimal (Locked)")

# --- Logic: Classification ---
def classify_with_gemini(keywords, api_key):
    client = genai.Client(api_key=api_key)
    results = []

    # Strictly defined system prompt
    system_instruction = """
    You are a professional SEO analyst. Label the provided keyword by the implied content format.
    Return ONLY the lowercase label from this specific list:
    - definition/factual
    - examples/list
    - comparison/pros-cons
    - asset/download/tool
    - product/service
    - instruction/how-to
    - consequence/effects/impacts
    - benefits/reason/justification
    - cost/price

    If not explicit or easily intuited, label 'unclear'.
    Do not provide any explanation, conversational text, or formatting other than the label itself.
    """

    progress_bar = st.progress(0)

    for i, kw in enumerate(keywords):
        try:
            # Using the corrected 2026 model ID and high-precision config
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,  # Zero randomness
                    thinking_config=types.ThinkingConfig(
                        thinking_level="minimal" # Low latency, high directness
                    )
                ),
                contents=f"Keyword: {kw}"
            )
            label = response.text.strip().lower()
            results.append(label)
        except Exception as e:
            results.append(f"Error: {str(e)}")

        progress_bar.progress((i + 1) / len(keywords))

    return results

# --- Main App Interface ---
uploaded_file = st.file_uploader("Upload Keyword List (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # User selects the correct column
    target_col = st.selectbox("Select the column containing keywords:", df.columns)

    if st.button("Run Classification"):
        if not gemini_api_key:
            st.warning("Please provide an API key in the sidebar.")
        else:
            with st.spinner("Classifying keywords..."):
                # Clean inputs to ensure they are strings
                keywords = df[target_col].astype(str).tolist()

                # Execute classification
                labels = classify_with_gemini(keywords, gemini_api_key)

                # Append results to dataframe
                df['Intent Label'] = labels

                st.success("Finished! See results below.")
                st.dataframe(df)

                # Download results
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results (.csv)", csv, "seo_intent_results.csv", "text/csv")
