import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional

# --- UI Configuration ---
st.set_page_config(page_title="SEO Classifier", layout="wide")
st.title("Classifier & Topic Suggester")

# --- Structured Output Definition ---
class IntentResult(BaseModel):
    index: int
    label: str
    topic: Optional[str] = "N/A"
    subtopic: Optional[str] = "N/A"

class BatchResponse(BaseModel):
    results: List[IntentResult]

# --- Session State Initialization ---
if "ai_suggestions" not in st.session_state:
    st.session_state.ai_suggestions = ""

# --- Logic: Topic Suggester ---
def suggest_topics(sample_keywords, api_key):
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Analyse these keywords and suggest 5-8 primary TOPICS and 10-15 granular SUBTOPICS.
    Keywords: {", ".join(sample_keywords)}

    Format your response clearly:
    TOPICS:
    (One per line)

    SUBTOPICS:
    (One per line)
    """
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            config=types.GenerateContentConfig(temperature=0.7),
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- Logic: Batch Classification ---
def classify_batches(keywords, api_key, custom_mode, topics="", subtopics=""):
    client = genai.Client(api_key=api_key)
    all_data = []
    batch_size = 30

    system_instruction = """
    You are a strict SEO analyst. Label intent: definition/factual, examples/list,
    comparison/pros-cons, asset/download/tool, product/service, instruction/how-to,
    consequence/effects/impacts, benefits/reason/justification, cost/price, unclear.
    """

    if custom_mode:
        system_instruction += f"\nTOPICS:\n{topics}\nSUBTOPICS:\n{subtopics}\n"
        system_instruction += "Assign one Topic and one Subtopic from the lists. Use 'N/A' if no fit."

    progress_bar = st.progress(0)
    for i in range(0, len(keywords), batch_size):
        chunk = keywords[i : i + batch_size]
        formatted = "\n".join([f"{i+idx}: {kw}" for idx, kw in enumerate(chunk)])

        try:
            res = client.models.generate_content(
                model="gemini-3-flash-preview",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=BatchResponse,
                    thinking_config=types.ThinkingConfig(thinking_level="minimal")
                ),
                contents=f"Classify:\n{formatted}"
            )
            for item in res.parsed.results:
                all_data.append({"Intent": item.label, "Topic": item.topic, "Subtopic": item.subtopic})
        except:
            for _ in chunk: all_data.append({"Intent": "error", "Topic": "error", "Subtopic": "error"})
        progress_bar.progress(min((i + batch_size) / len(keywords), 1.0))
    return all_data

# --- Sidebar: Configuration & Custom Inputs ---
with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password")
    use_custom = st.checkbox("Enable Custom Categorisation")

    # These variables need to be initialized so the main app can read them even if hidden
    topics_area = ""
    subtopics_area = ""

    # HIDE/SHOW LOGIC: Only show the form if the checkbox is True
    if use_custom:
        with st.form("category_form"):
            st.write("### Provide topics/subtopics here. If you don't do this, the tool will only generate intents")
            topics_area = st.text_area("Primary Topics (Required)", height=150, help="Paste topics here.")
            subtopics_area = st.text_area("Subtopics (Optional)", height=150, help="Paste subtopics here.")
            st.form_submit_button("Save")

# --- Main App Interface ---
uploaded_file = st.file_uploader("Upload Keyword CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    target_col = st.selectbox("Keyword Column", df.columns)

    if use_custom:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✨ Generate AI Topic Suggestions"):
                if not api_key:
                    st.error("Please enter an API Key first.")
                else:
                    with st.spinner("Analysing keyword sample..."):
                        sample = df[target_col].sample(n=min(50, len(df))).astype(str).tolist()
                        st.session_state.ai_suggestions = suggest_topics(sample, api_key)

        with col2:
            if st.button("🗑️ Clear Suggestions"):
                st.session_state.ai_suggestions = ""
                st.rerun()

        if st.session_state.ai_suggestions:
            st.markdown("---")
            st.markdown("### AI Recommended Topics & Subtopics")
            st.info("Copy these and paste them into the sidebar fields. Click 'Save Strategy' to lock them in.")
            st.code(st.session_state.ai_suggestions)
            st.markdown("---")

    if st.button("Run Full Classification"):
        if not api_key:
            st.error("Missing API Key.")
        elif use_custom and not topics_area:
            st.error("Please provide topics in the sidebar form.")
        else:
            with st.spinner("Classifying in batches..."):
                results = classify_batches(df[target_col].tolist(), api_key, use_custom, topics_area, subtopics_area)
                res_df = pd.DataFrame(results)
                df['Intent'] = res_df['Intent']
                if use_custom:
                    df['Topic'], df['Subtopic'] = res_df['Topic'], res_df['Subtopic']

                st.success("Complete!")
                st.dataframe(df)
                st.download_button("📥 Download Results", df.to_csv(index=False), "results.csv", "text/csv")
