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
    funnel_stage: str
    confidence: float
    topic: Optional[str] = "N/A"
    subtopic: Optional[str] = "N/A"

class BatchResponse(BaseModel):
    results: List[IntentResult]

# --- Session State Initialization ---
if "ai_suggestions" not in st.session_state:
    st.session_state.ai_suggestions = ""
if "topics" not in st.session_state:
    st.session_state.topics = ""
if "subtopics" not in st.session_state:
    st.session_state.subtopics = ""

# --- Logic: Topic Suggester ---
def suggest_topics(sample_keywords, api_key):
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Analyse these keywords. Provide:
    1. A list of primary TOPICS.
    2. A list of deduplicated, concise SUBTOPIC 'stems' (up to 5 per topic).
    Keywords: {", ".join(sample_keywords)}

    Output as two clean blocks for copy-pasting:
    --- TOPICS BLOCK ---
    (One per line, no symbols)

    --- SUBTOPICS BLOCK ---
    (One per line, no symbols)
    """
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            config=types.GenerateContentConfig(temperature=0.0),
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
    You are a strict SEO analyst. For each keyword:
    1. Label intent: definition/factual, examples/list, comparison/pros-cons, asset/download/tool, product/service, instruction/how-to, consequence/effects/impacts, benefits/reason/justification, cost/price, unclear.
    2. Map to Marketing Funnel stage: Awareness, Consideration, or Transactional.
    3. Provide a confidence score between 0.0 and 1.0 for your classification.
    """

    if custom_mode:
        system_instruction += f"\nTOPICS:\n{topics}\nSUBTOPICS:\n{subtopics}\n"
        system_instruction += "\nAssign one Topic and one Subtopic from the lists provided. "
        system_instruction += "Use 'N/A' if no fit."

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
                all_data.append({
                    "Intent": item.label, 
                    "Funnel": item.funnel_stage,
                    "Confidence": item.confidence,
                    "Topic": item.topic, 
                    "Subtopic": item.subtopic
                })
        except:
            for _ in chunk: all_data.append({
                "Intent": "error", "Funnel": "error", "Confidence": 0.0, "Topic": "error", "Subtopic": "error"
            })
        progress_bar.progress(min((i + batch_size) / len(keywords), 1.0))
    return all_data

# --- Sidebar: Configuration & Custom Inputs ---
with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password")
    use_custom = st.checkbox("Enable Custom Categorisation")

    # HIDE/SHOW LOGIC: Only show the form if the checkbox is True
    if use_custom:
        with st.form("category_form"):
            st.write("### Provide your classification strategy here.")
            # Use session state to persist the values
            st.session_state.topics = st.text_area("Primary Topics (Required)", 
                                        value=st.session_state.topics,
                                        height=150, 
                                        help="Paste primary topics here (one per line).")
            st.session_state.subtopics = st.text_area("Subtopics (Optional)", 
                                        value=st.session_state.subtopics,
                                        height=150, 
                                        help="Paste granular subtopics here (one per line).")
            st.form_submit_button("Save Strategy")

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
                        sample = df[target_col].sample(n=min(150, len(df))).astype(str).tolist()
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
        elif use_custom and not st.session_state.topics:
            st.error("Please provide topics in the sidebar form.")
        else:
            with st.spinner("Classifying in batches..."):
                results = classify_batches(df[target_col].tolist(), api_key, use_custom, 
                                           st.session_state.topics, st.session_state.subtopics)
                res_df = pd.DataFrame(results)
                df['Intent'] = res_df['Intent']
                df['Funnel'] = res_df['Funnel']
                df['Confidence'] = res_df['Confidence']
                if use_custom:
                    df['Topic'], df['Subtopic'] = res_df['Topic'], res_df['Subtopic']

                st.success("Complete!")
                st.dataframe(df)
                st.download_button("📥 Download Results", df.to_csv(index=False), "results.csv", "text/csv")
