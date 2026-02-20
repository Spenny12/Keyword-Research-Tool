import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional
import random

# --- UI Configuration ---
st.set_page_config(page_title="Pro SEO Classifier", layout="wide")
st.title("🚀 Pro SEO Classifier & Topic Suggester")

# --- Structured Output Definition ---
class IntentResult(BaseModel):
    index: int
    label: str
    topic: Optional[str] = "N/A"
    subtopic: Optional[str] = "N/A"

class BatchResponse(BaseModel):
    results: List[IntentResult]

# --- Logic: Topic Suggester ---
def suggest_topics(sample_keywords, api_key):
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Analyze these keywords and suggest a list of 5-8 high-level TOPICS
    and a list of 10-15 granular SUBTOPICS that could categorize them.

    Keywords: {", ".join(sample_keywords)}

    Format your response exactly like this:
    TOPICS:
    Topic 1
    Topic 2

    SUBTOPICS:
    Subtopic 1
    Subtopic 2
    """
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            config=types.GenerateContentConfig(temperature=0.7), # Higher temp for brainstorming
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating suggestions: {e}"

# --- Logic: Batch Classification (Locked at 0.0 Temperature) ---
def classify_batches(keywords, api_key, custom_mode, topics="", subtopics=""):
    client = genai.Client(api_key=api_key)
    all_data = []
    batch_size = 30

    system_instruction = """
    You are a strict SEO bot. Label keywords by intent (definition/factual, examples/list,
    comparison/pros-cons, asset/download/tool, product/service, instruction/how-to,
    consequence/effects/impacts, benefits/reason/justification, cost/price, unclear).
    """

    if custom_mode:
        system_instruction += f"\nTOPICS:\n{topics}\nSUBTOPICS:\n{subtopics}\n"
        system_instruction += "Assign exactly one Topic and one Subtopic. Use 'N/A' if no fit."

    progress_bar = st.progress(0)
    for i in range(0, len(keywords), batch_size):
        chunk = keywords[i : i + batch_size]
        formatted = "\n".join([f"{i+idx}: {kw}" for idx, kw in enumerate(chunk)])

        try:
            res = client.models.generate_content(
                model="gemini-3-flash-preview",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0, # High precision
                    response_mime_type="application/json",
                    response_schema=BatchResponse
                ),
                contents=f"Classify:\n{formatted}"
            )
            for item in res.parsed.results:
                all_data.append({"Intent": item.label, "Topic": item.topic, "Subtopic": item.subtopic})
        except:
            for _ in chunk: all_data.append({"Intent": "error", "Topic": "error", "Subtopic": "error"})
        progress_bar.progress(min((i + batch_size) / len(keywords), 1.0))
    return all_data

# --- Sidebar ---
with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password")
    use_custom = st.checkbox("Enable Custom Categorisation")

    topics_input = ""
    subtopics_input = ""

    if use_custom:
        st.markdown("### Topic Strategy")
        topics_input = st.text_area("Primary Topics (Required)", placeholder="E.g. SEO, Content, Technical")
        subtopics_input = st.text_area("Subtopics (Optional)", placeholder="E.g. Backlinks, Site Speed")

# --- Main App ---
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    target_col = st.selectbox("Keyword Column", df.columns)

    # NEW: The Suggester UI
    if use_custom:
        if st.button("✨ Suggest Topics from My Data"):
            if not api_key:
                st.error("Enter API Key first.")
            else:
                # Randomly sample 50 keywords for the AI to "read"
                sample = df[target_col].sample(n=min(50, len(df))).astype(str).tolist()
                suggestions = suggest_topics(sample, api_key)
                st.markdown("### 💡 AI Recommendations")
                st.info("Copy and paste these into the text boxes in the sidebar.")
                st.code(suggestions)

    if st.button("Run Full Classification"):
        if not api_key: st.error("Missing API Key.")
        else:
            results = classify_batches(df[target_col].tolist(), api_key, use_custom, topics_input, subtopics_input)
            res_df = pd.DataFrame(results)
            df['Intent'] = res_df['Intent']
            if use_custom:
                df['Topic'], df['Subtopic'] = res_df['Topic'], res_df['Subtopic']

            st.success("Complete!")
            st.dataframe(df)
            st.download_button("📥 Download CSV", df.to_csv(index=False), "results.csv", "text/csv")
