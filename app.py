import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- UI Configuration ---
st.set_page_config(page_title="SEO Classifier", layout="wide")

# --- Structured Output Definitions ---
class IntentResult(BaseModel):
    idx: int
    i: str  # intent code
    f: str  # funnel code

class TopicResult(BaseModel):
    idx: int
    t: str # topic
    s: str # subtopic

class IntentBatchResponse(BaseModel):
    results: List[IntentResult]

class TopicBatchResponse(BaseModel):
    results: List[TopicResult]

# --- Session State Initialization ---
if "ai_suggestions" not in st.session_state:
    st.session_state.ai_suggestions = ""
if "topics" not in st.session_state:
    st.session_state.topics = ""
if "subtopics" not in st.session_state:
    st.session_state.subtopics = ""

# --- Logic: Topic Suggester ---
def suggest_topics(sample_keywords, api_key):
    system_instruction = "You are a technical SEO specialist. Provide a CONCISE list of topics. Output ONLY the requested blocks."
    limit_text = "Aim for a maximum of 5 primary TOPICS and up to 15 subtopics."

    prompt = f"""
    Analyse these keywords and provide:
    1. Primary TOPICS ({limit_text}).
    2. Deduplicated, concise SUBTOPIC 'stems'.

    Keywords:
    {'\n'.join(sample_keywords)}

    Output EXACTLY in this format:
    --- TOPICS BLOCK ---
    (Topic Name 1)
    (Topic Name 2)

    --- SUBTOPICS BLOCK ---
    (Subtopic Stem 1)
    (Subtopic Stem 2)
    (Subtopic Stem 3)

    Rules:
    - Focus on high-level SEO themes for Topics.
    - Focus on granular user intent/product variations for Subtopics.
    - One per line.
    - No descriptions, no colons, no bolding, no introductory text.
    """
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.0),
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- Logic: Batch Processing ---
def process_batches(keywords, api_key, mode, topics="", subtopics=""):
    model_id = "gemini-3-flash-preview"
    batch_size = 125
    max_workers = 5
    
    intent_map = {
        "1": "definition/factual", "2": "examples/list", "3": "comparison/pros-cons",
        "4": "asset/download/tool", "5": "product/service", "6": "instruction/how-to",
        "7": "consequence/effects/impacts", "8": "benefits/reason/justification",
        "9": "cost/price", "10": "unclear"
    }
    funnel_map = {"A": "Awareness", "C": "Consideration", "T": "Transactional"}

    if mode == "intent":
        system_instruction = f"SEO Intent: {','.join([f'{k}:{v}' for k,v in intent_map.items()])}|A:Aware,C:Consid,T:Trans. Output JSON."
        schema = IntentBatchResponse
    else:
        c_topics = ",".join([t.strip() for t in topics.split("\n") if t.strip()])
        c_subtopics = ",".join([s.strip() for s in subtopics.split("\n") if s.strip()])
        system_instruction = f"Classify keywords into Topics: [{c_topics}] and Subtopics: [{c_subtopics}]. Return ONLY the Topic and Subtopic names exactly as provided in the lists. Output JSON."
        schema = TopicBatchResponse

    final_results = [None] * len(keywords)
    
    status_text = st.empty()
    progress_bar = st.progress(0.0)
    
    chunks = []
    for i in range(0, len(keywords), batch_size):
        chunks.append([(j, keywords[j]) for j in range(i, min(i + batch_size, len(keywords)))])

    total_chunks = len(chunks)
    completed_chunks = 0
    
    def process_chunk(chunk):
        chunk_out = []
        try:
            formatted = "\n".join([f"{idx_in_batch}|{kw}" for idx_in_batch, (global_idx, kw) in enumerate(chunk)])
            mapping = {idx_in_batch: global_idx for idx_in_batch, (global_idx, kw) in enumerate(chunk)}

            client = genai.Client(api_key=api_key)
            res = client.models.generate_content(
                model=model_id,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.0,
                ),
                contents=formatted
            )
            parsed = res.parsed

            if parsed and parsed.results:
                for item in parsed.results:
                    g_idx = mapping.get(item.idx)
                    if g_idx is not None:
                        if mode == "intent":
                            data = {"Intent": intent_map.get(item.i, "unclear"), "Funnel": funnel_map.get(item.f.upper(), "Awareness")}
                        else:
                            data = {"Topic": item.t, "Subtopic": item.s}
                        chunk_out.append((g_idx, data))
        except Exception: pass
        return chunk_out

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk, c): c for c in chunks}
        for future in as_completed(futures):
            results = future.result()
            for g_idx, data in results:
                final_results[g_idx] = data

            completed_chunks += 1
            progress_bar.progress(completed_chunks / total_chunks)
            status_text.text(f"Processed batch {completed_chunks} of {total_chunks}...")

    default_data = {"Intent": "error", "Funnel": "N/A"} if mode == "intent" else {"Topic": "error", "Subtopic": "error"}
    return [r if r is not None else default_data for r in final_results]

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Readme", "1. Intent Classifier", "2. Topic Mapper"])

st.sidebar.markdown("---")
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API Key. Speak to Tom if you don't have one.")

if page == "Readme":
    st.title("📖 How to Use This Tool")
    st.markdown("""
    ### Workflow Overview
    Follow these steps to classify your keywords accurately and efficiently:

    1.  **Gather Keywords:** Prepare a clean list of search queries in a `.csv` file. Remove unnecessary data like search volumes to keep it simple.
    2.  **API Keys:** Speak to Tom to obtain a valid Gemini API key.
    3.  **Step 1 - Intent Classification:**
        *   Navigate to **'1. Intent Classifier'**.
        *   Upload your CSV and select the keyword column.
        *   Run the classifier and download `intent_results.csv`.
    4.  **Step 2 - Topic Mapping:**
        *   Navigate to **'2. Topic Mapper'**.
        *   Upload the file from Step 1.
        *   **Generate AI Suggestions** (optional) to build your strategy.
        *   Review and edit the topics in the sidebar.
        *   Run Topic Mapping and export the final report.
    """)
    st.info("💡 **Pro Tip:** Splitting the process into two steps ensures higher accuracy and prevents timeouts on large datasets.")

elif page == "1. Intent Classifier":
    st.title("Step 1: Intent & Funnel Classifier")
    st.info("Classify Search Intent and Marketing Funnel stage using Gemini AI.")
    
    uploaded_file = st.file_uploader("Upload Keyword CSV", type=["csv"], key="intent_upload")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_col = st.selectbox("Keyword Column", df.columns, key="intent_col")
        
        if st.button("Run Intent Classification"):
            if not api_key:
                st.error("Missing Gemini API Key.")
            else:
                with st.spinner("Classifying..."):
                    raw_kws = df[target_col].astype(str).tolist()
                    unique_kws = list(dict.fromkeys([kw for kw in raw_kws if kw.strip() and kw.lower() != 'nan']))
                    results = process_batches(unique_kws, api_key, mode="intent")
                    
                    results_map = dict(zip(unique_kws, results))
                    final_results = [results_map.get(kw, {"Intent": "N/A", "Funnel": "N/A"}) for kw in raw_kws]
                    
                    res_df = pd.DataFrame(final_results)
                    df['Intent'], df['Funnel'] = res_df['Intent'], res_df['Funnel']
                    
                    st.success("Complete!")
                    st.dataframe(df)
                    st.download_button("📥 Download Intent Results", df.to_csv(index=False), "intent_results.csv", "text/csv")

else:
    st.title("Step 2: Custom Topic Mapper")
    st.info("Add custom Topic and Subtopic categorisation using your predefined strategy.")
    
    with st.sidebar:
        st.markdown("---")
        st.write("### Classification Strategy")
        st.session_state.topics = st.text_area("Primary Topics", value=st.session_state.topics, height=150)
        st.session_state.subtopics = st.text_area("Subtopics (Optional)", value=st.session_state.subtopics, height=150)

    uploaded_file = st.file_uploader("Upload Intent CSV", type=["csv"], key="topic_upload")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_col = st.selectbox("Keyword Column", df.columns, key="topic_col")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✨ Generate AI Suggestions"):
                if not api_key: st.error("Missing Gemini API Key.")
                else:
                    with st.spinner("Analysing sample..."):
                        unique_kws = [kw for kw in df[target_col].astype(str).unique() if kw.strip().lower() not in ['nan', 'null', '']]
                        sample = pd.Series(unique_kws).sample(n=min(150, len(unique_kws))).tolist()
                        st.session_state.ai_suggestions = suggest_topics(sample, api_key)
        with col2:
            if st.button("🗑️ Clear Suggestions"):
                st.session_state.ai_suggestions = ""
                st.rerun()

        if st.session_state.ai_suggestions:
            st.code(st.session_state.ai_suggestions)

        if st.button("Run Topic Mapping"):
            if not api_key: st.error("Missing Gemini API Key.")
            elif not st.session_state.topics: st.error("Provide topics in the sidebar.")
            else:
                with st.spinner("Mapping topics..."):
                    raw_kws = df[target_col].astype(str).tolist()
                    unique_kws = list(dict.fromkeys([kw for kw in raw_kws if kw.strip() and kw.lower() != 'nan']))
                    results = process_batches(unique_kws, api_key, mode="topic", 
                                           topics=st.session_state.topics, subtopics=st.session_state.subtopics)
                    
                    results_map = dict(zip(unique_kws, results))
                    final_results = [results_map.get(kw, {"Topic": "N/A", "Subtopic": "N/A"}) for kw in raw_kws]
                    
                    res_df = pd.DataFrame(final_results)
                    df['Topic'], df['Subtopic'] = res_df['Topic'], res_df['Subtopic']
                    
                    st.success("Complete!")
                    st.dataframe(df)
                    st.download_button("📥 Download Final Results", df.to_csv(index=False), "final_seo_results.csv", "text/csv")
