import streamlit as st
import pandas as pd
from google import genai
from google.genai import types

# --- UI Configuration ---
st.set_page_config(page_title="SEO Intent Classifier", layout="wide")
st.title("Keyword Intent Classifier")

# --- Sidebar ---
with st.sidebar:
    gemini_api_key = st.text_input("Enter Gemini API Key", type="password")
    st.info("Using Gemini 3 Flash for high-speed categorisation.")
    temperature = st.slider("Precision (0 = Strict, 1 = Creative)", 0.0, 1.0, 0.1)

# --- Logic ---
def classify_with_gemini(keywords, api_key):
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Client: {e}")
        return ["Client Error"] * len(keywords)

    results = []
    progress_bar = st.progress(0)

    for i, kw in enumerate(keywords):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash",
                config=types.GenerateContentConfig(
                    system_instruction="Label the keyword intent.", # Shortened for test
                    temperature=0.1
                ),
                contents=f"Keyword: {kw}"
            )
            results.append(response.text.strip().lower())
        except Exception as e:
            # THIS IS THE IMPORTANT PART:
            # It will print the actual error message to your Streamlit screen
            st.error(f"Error on keyword '{kw}': {e}")
            results.append(f"Error: {type(e).__name__}")

        progress_bar.progress((i + 1) / len(keywords))
    return results

# --- File Handling ---
uploaded_file = st.file_uploader("Upload Keyword CSV", type=["csv"])

if uploaded_file:
    # No openpyxl engine needed for read_csv
    df = pd.read_csv(uploaded_file)

    st.write("### Data Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("Select the keyword column:", df.columns)

    if st.button("Categorize Keywords"):
        if not gemini_api_key:
            st.error("Missing API Key.")
        else:
            with st.spinner("Processing..."):
                keywords = df[target_col].astype(str).tolist()
                df['Intent Label'] = classify_with_gemini(keywords, gemini_api_key)

                st.success("Done!")
                st.dataframe(df)

                # Direct CSV Export
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results",
                    data=csv_data,
                    file_name="classified_keywords.csv",
                    mime="text/csv"
                )
