import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import random

# --- UI Configuration ---
st.set_page_config(page_title="SEO Classifier", layout="wide")

# --- Structured Output Definitions ---
class IntentResult(BaseModel):
    idx: int = Field(description="The exact index number provided in the prompt")
    intent: str = Field(default="10", description="The numeric intent code (e.g., '1', '2')")
    funnel: str = Field(default="A", description="The single-letter funnel code (e.g., 'A', 'C', 'T')")

class TopicResult(BaseModel):
    idx: int = Field(description="The exact index number provided in the prompt")
    topic: str = Field(default="N/A", description="The assigned primary topic exactly as provided, or 'N/A' if none fit")

class SubtopicResult(BaseModel):
    idx: int = Field(description="The exact index number provided in the prompt")
    subtopic: str = Field(default="N/A", description="The assigned subtopic exactly as provided, or 'N/A' if none fit")

class IntentBatchResponse(BaseModel):
    results: List[IntentResult]

class TopicBatchResponse(BaseModel):
    results: List[TopicResult]

class SubtopicBatchResponse(BaseModel):
    results: List[SubtopicResult]

# --- Session State Initialization ---
if "ai_suggestions" not in st.session_state:
    st.session_state.ai_suggestions = ""
if "topics" not in st.session_state:
    st.session_state.topics = ""
if "subtopics" not in st.session_state:
    st.session_state.subtopics = ""
if "working_df" not in st.session_state:
    st.session_state.working_df = pd.DataFrame()
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = ""

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
            model="gemini-3.1-flash-lite",
            config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.0),
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- Logic: Batch Processing ---
def process_batches(keywords, api_key, mode, topics="", subtopics=""):
    model_id = "gemini-3.1-flash-lite"
    batch_size = 50
    max_workers = 2

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

    elif mode == "topic":
        # Formatted as a vertical bulleted list to prevent the LLM from dropping the tail end
        c_topics = "\n".join([f"- {t.strip()}" for t in topics.split("\n") if t.strip()])
        system_instruction = f"""
        Classify keywords into one of the following Primary Topics:
        {c_topics}

        STRICT RULES:
        1. Read the ENTIRE list of topics above.
        2. Return ONLY the Topic name exactly as provided in the list.
        3. If no topic is a good fit, return 'N/A'.
        4. Accuracy is more important than coverage.
        Output JSON.
        """
        schema = TopicBatchResponse

    elif mode == "subtopic":
        # Formatted as a vertical bulleted list to prevent the LLM from dropping the tail end
        c_subtopics = "\n".join([f"- {s.strip()}" for s in subtopics.split("\n") if s.strip()])
        system_instruction = f"""
        Classify keywords into one of the following Subtopics:
        {c_subtopics}

        STRICT RULES:
        1. Read the ENTIRE list of subtopics above.
        2. Only assign a Subtopic if it is HIGHLY RELEVANT to the keyword.
        3. Return ONLY the Subtopic name exactly as provided in the list.
        4. If no subtopic in the list is a good fit, return 'N/A'.
        Output JSON.
        """
        schema = SubtopicBatchResponse

    if not keywords:
        return []

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
        formatted = "\n".join([f"{idx_in_batch}|{kw}" for idx_in_batch, (global_idx, kw) in enumerate(chunk)])
        mapping = {idx_in_batch: global_idx for idx_in_batch, (global_idx, kw) in enumerate(chunk)}

        for attempt in range(4):
            try:
                if attempt > 0:
                    sleep_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(sleep_time)
                else:
                    time.sleep(random.uniform(0.1, 0.5))

                client = genai.Client(api_key=api_key)
                res = client.models.generate_content(
                    model=model_id,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction + "\nIMPORTANT: You must return a JSON object with a 'results' key containing a list of objects. Each object MUST have an 'idx' key (matching the index provided) and the appropriate classification keys.",
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
                                safe_intent = str(item.intent).strip()
                                safe_funnel = str(item.funnel).strip().upper()
                                data = {"Intent": intent_map.get(safe_intent, "unclear"), "Funnel": funnel_map.get(safe_funnel, "Awareness")}
                            elif mode == "topic":
                                data = {"Topic": item.topic}
                            elif mode == "subtopic":
                                data = {"Subtopic": item.subtopic}
                            chunk_out.append((g_idx, data))
                    return chunk_out
            except Exception as e:
                err_msg = str(e)
                code_match = re.search(r'\b(4\d{2}|5\d{2})\b', err_msg)
                err_code = f" {code_match.group(1)}" if code_match else ""
                err_type = type(e).__name__

                if attempt == 3:
                    print(f"CRITICAL: Chunk error after 4 attempts: [{err_type}{err_code}] {err_msg}")
                    for global_idx, kw in chunk:
                        display_err = f"ERR: {err_type}{err_code}"
                        if mode == "intent":
                            error_data = {"Intent": display_err, "Funnel": "N/A"}
                        elif mode == "topic":
                            error_data = {"Topic": display_err}
                        else:
                            error_data = {"Subtopic": display_err}
                        chunk_out.append((global_idx, error_data))
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

    if mode == "intent":
        default_data = {"Intent": "error", "Funnel": "N/A"}
    elif mode == "topic":
        default_data = {"Topic": "error"}
    else:
        default_data = {"Subtopic": "error"}

    return [r if r is not None else default_data for r in final_results]

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Readme", "1. Intent Classifier", "2. Topic Mapper"])

st.sidebar.markdown("---")
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API Key. Speak to Tom if you don't have one.")

if page == "Readme":
    st.title("💡 How to Use This Tool")
    st.markdown("""
    ### Workflow Overview
    Follow these steps to classify your keywords accurately and efficiently:

    1.  **Gather Keywords:** Prepare a clean list of search queries in a `.csv` file. Remove unnecessary data like search volumes to keep it simple.
    2.  **API Keys:** Speak to Tom to obtain a valid Gemini API key.
    3.  **Step 1 - Intent Classification:**
        * Navigate to **'1. Intent Classifier'**.
        * Upload your CSV and select the keyword column.
        * Run the classifier and download `intent_results.csv`.
    4.  **Step 2 - Topic Mapping:**
        * Navigate to **'2. Topic Mapper'**.
        * Upload the file from Step 1.
        * Run Topics and Subtopics completely independently of each other using the separate buttons provided to ensure high accuracy.
        * Export the final report.
    """)
    st.info("🎯 **Pro Tip:** Running Topics and Subtopics separately prevents the AI from getting confused and blending categories together.")

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
                    st.download_button("⬇️ Download Intent Results", df.to_csv(index=False), "intent_results.csv", "text/csv")

else:
    st.title("Step 2: Custom Topic Mapper")
    st.info("Map your custom Topics and Subtopics independently to ensure maximum accuracy.")

    with st.sidebar:
        st.markdown("---")
        st.write("### Classification Strategy")
        st.session_state.topics = st.text_area("Primary Topics", value=st.session_state.topics, height=150)
        st.session_state.subtopics = st.text_area("Subtopics", value=st.session_state.subtopics, height=150)

    uploaded_file = st.file_uploader("Upload CSV (e.g. from Step 1)", type=["csv"], key="topic_upload")
    if uploaded_file:
        # Save to session state so UI reruns don't erase the progress
        if st.session_state.last_uploaded != uploaded_file.name:
            st.session_state.working_df = pd.read_csv(uploaded_file)
            st.session_state.last_uploaded = uploaded_file.name

        df = st.session_state.working_df
        target_col = st.selectbox("Keyword Column", df.columns, key="topic_col")

        with st.expander("✨ AI Strategy Suggester (Optional)"):
            st.write("Need help building a list of Topics/Subtopics? Generate a sample strategy based on your data.")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Generate AI Suggestions"):
                    if not api_key: st.error("Missing Gemini API Key.")
                    else:
                        with st.spinner("Analysing sample..."):
                            unique_kws = [kw for kw in df[target_col].astype(str).unique() if kw.strip().lower() not in ['nan', 'null', '']]
                            sample = pd.Series(unique_kws).sample(n=min(200, len(unique_kws))).tolist()
                            st.session_state.ai_suggestions = suggest_topics(sample, api_key)
            with col2:
                if st.button("🗑️ Clear Suggestions"):
                    st.session_state.ai_suggestions = ""
                    st.rerun()

            if st.session_state.ai_suggestions:
                st.code(st.session_state.ai_suggestions)

        st.markdown("### Execution")
        st.write("Run these sequentially. Data is saved in memory so you won't lose your progress between steps.")

        exec_col1, exec_col2 = st.columns(2)

        with exec_col1:
            if st.button("1️⃣ Run Topics ONLY", use_container_width=True):
                if not api_key: st.error("Missing Gemini API Key.")
                elif not st.session_state.topics: st.error("Provide topics in the sidebar.")
                else:
                    with st.spinner("Mapping topics..."):
                        raw_kws = df[target_col].astype(str).tolist()
                        unique_kws = list(dict.fromkeys([kw for kw in raw_kws if kw.strip() and kw.lower() != 'nan']))
                        results = process_batches(unique_kws, api_key, mode="topic", topics=st.session_state.topics)

                        results_map = dict(zip(unique_kws, results))
                        final_results = [results_map.get(kw, {"Topic": "N/A"}) for kw in raw_kws]

                        df['Topic'] = pd.DataFrame(final_results)['Topic']
                        st.session_state.working_df = df
                        st.success("Topics Mapped!")
                        st.rerun()

        with exec_col2:
            if st.button("2️⃣ Run Subtopics ONLY", use_container_width=True):
                if not api_key: st.error("Missing Gemini API Key.")
                elif not st.session_state.subtopics: st.error("Provide subtopics in the sidebar.")
                else:
                    with st.spinner("Mapping subtopics..."):
                        raw_kws = df[target_col].astype(str).tolist()
                        unique_kws = list(dict.fromkeys([kw for kw in raw_kws if kw.strip() and kw.lower() != 'nan']))
                        results = process_batches(unique_kws, api_key, mode="subtopic", subtopics=st.session_state.subtopics)

                        results_map = dict(zip(unique_kws, results))
                        final_results = [results_map.get(kw, {"Subtopic": "N/A"}) for kw in raw_kws]

                        df['Subtopic'] = pd.DataFrame(final_results)['Subtopic']
                        st.session_state.working_df = df
                        st.success("Subtopics Mapped!")
                        st.rerun()

        st.markdown("### Current Results")
        st.dataframe(st.session_state.working_df)
        st.download_button("⬇️ Download Current Results", st.session_state.working_df.to_csv(index=False), "current_seo_results.csv", "text/csv")
