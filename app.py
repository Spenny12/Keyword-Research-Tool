import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List

# --- UI Configuration ---
st.set_page_config(page_title="Candour SEO Classifier", layout="wide")
st.title("SEO Intent Classifier")

# --- Structured Output Definition ---
class IntentResult(BaseModel):
    index: int
    label: str

class BatchResponse(BaseModel):
    results: List[IntentResult]

# --- Sidebar ---
with st.sidebar:
    gemini_api_key = st.text_input("Enter Gemini API Key", type="password")
    st.divider()

# --- Logic: Batch Processing ---
def classify_batches(keywords, api_key):
    client = genai.Client(api_key=api_key)
    all_labels = [None] * len(keywords)
    batch_size = 50

    system_instruction = """
    You are a strict SEO classification bot.
    Label each keyword by the implied content format.
    Return only labels from this list:
    - definition/factual, examples/list, comparison/pros-cons, asset/download/tool,
      product/service, instruction/how-to, consequence/effects/impacts,
      benefits/reason/justification, cost/price, unclear.
    """

    progress_bar = st.progress(0)

    # Split keywords into chunks of 30
    for i in range(0, len(keywords), batch_size):
        chunk = keywords[i : i + batch_size]

        # Create a numbered list for the prompt to ensure tracking
        formatted_list = "\n".join([f"{i + idx}: {kw}" for idx, kw in enumerate(chunk)])

        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=BatchResponse, # Forces structured list
                ),
                contents=f"Classify these keywords by their index:\n{formatted_list}"
            )

            # Parse the JSON response directly into our results list
            batch_data = response.parsed
            for item in batch_data.results:
                if item.index < len(all_labels):
                    all_labels[item.index] = item.label.strip().lower()

        except Exception as e:
            st.error(f"Batch Error starting at index {i}: {e}")
            # Fill failed slots with error
            for idx in range(i, min(i + batch_size, len(keywords))):
                if all_labels[idx] is None:
                    all_labels[idx] = "error"

        progress_bar.progress(min((i + batch_size) / len(keywords), 1.0))

    return all_labels

# --- Main App Interface ---
uploaded_file = st.file_uploader("Upload Keyword CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"### File Loaded ({len(df)} keywords)")

    target_col = st.selectbox("Select Keyword Column:", df.columns)

    if st.button("Run Batch Classification"):
        if not gemini_api_key:
            st.error("Missing API Key.")
        else:
            with st.spinner(f"Processing {len(df)} keywords in batches..."):
                keywords = df[target_col].astype(str).tolist()
                df['Intent Label'] = classify_batches(keywords, gemini_api_key)

                st.success("Batch Processing Complete!")
                st.dataframe(df)

                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Classified CSV", csv_data, "batch_results.csv", "text/csv")
