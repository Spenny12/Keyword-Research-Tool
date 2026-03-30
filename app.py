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
    client = genai.Client(api_key=api_key)
    system_instruction = "You are a technical SEO specialist. Analyse keywords to provide structured topic and subtopic recommendations. Use British English and be concise."
    prompt = f"""
    Analyse these keywords and provide:
    1. A list of primary TOPICS.
    2. A list of deduplicated, concise SUBTOPIC 'stems' (up to 3 per topic).

    Keywords:
    {'\n'.join(sample_keywords)}

    Output as two clean blocks for copy-pasting:
    --- TOPICS BLOCK ---
    (One per line, no symbols)

    --- SUBTOPICS BLOCK ---
    (One per line, no symbols)
    """
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0
            ),
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- Logic: Generic Batch Processor ---
def process_batches(keywords, api_key, mode, topics="", subtopics=""):
    client = genai.Client(api_key=api_key)
    model_id = "gemini-3-flash-preview"
    batch_size = 100 
    max_workers = 10 
    
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
        system_instruction = f"Classify keywords into Topics: [{c_topics}] and Subtopics: [{c_subtopics}]. Use exact matches if possible. Output JSON."
        schema = TopicBatchResponse

    # Context Caching
    cache = None
    try:
        cache = client.caches.create(
            model=model_id,
            config=types.CacheConfig(system_instruction=system_instruction, ttl_seconds=3600)
        )
    except Exception: pass

    final_results = [None] * len(keywords)
    keyword_status = [{"id": i, "kw": kw, "status": "pending"} for i, kw in enumerate(keywords)]
    
    pass_count = 1
    status_text = st.empty()
    progress_bar = st.progress(0)

    while any(s["status"] in ["pending", "retry_skip"] for s in keyword_status):
        to_process = [s for s in keyword_status if s["status"] in ["pending", "retry_skip"]]
        if not to_process or pass_count > 15: break
        if pass_count > 1: time.sleep(1)

        chunks = [to_process[i : i + batch_size] for i in range(0, len(to_process), batch_size)]
        total_chunks = len(chunks)
        completed_chunks = 0
        status_text.text(f"Pass {pass_count}: Processing {len(to_process)} keywords...")

        def process_chunk(chunk):
            mapping = {idx: item["id"] for idx, item in enumerate(chunk)}
            formatted = "\n".join([f"{idx}|{item['kw']}" for idx, item in enumerate(chunk)])
            chunk_results = []
            try:
                res = client.models.generate_content(
                    model=model_id,
                    config=types.GenerateContentConfig(
                        system_instruction=None if cache else system_instruction,
                        cached_content=cache.name if cache else None,
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.0,
                    ),
                    contents=formatted
                )
                
                batch_data = {}
                if res.parsed and res.parsed.results:
                    for item in res.parsed.results:
                        if mode == "intent":
                            batch_data[item.idx] = {
                                "Intent": intent_map.get(item.i, "unclear"), 
                                "Funnel": funnel_map.get(item.f.upper(), "Awareness")
                            }
                        else:
                            batch_data[item.idx] = {"Topic": item.t, "Subtopic": item.s}
                
                for idx in range(len(chunk)):
                    global_idx = mapping[idx]
                    item = batch_data.get(idx)
                    if item:
                        chunk_results.append({"global_id": global_idx, "data": item, "new_status": "completed"})
                    else:
                        new_status = "failed_permanent" if keyword_status[global_idx]["status"] == "retry_skip" else "retry_skip"
                        chunk_results.append({"global_id": global_idx, "data": None, "new_status": new_status})
            except Exception as e:
                error_msg = str(e).lower()
                is_server_error = any(code in error_msg for code in ["503", "500", "504", "overloaded", "deadline", "timeout"])
                for item in chunk:
                    chunk_results.append({
                        "global_id": item["id"], 
                        "data": None, 
                        "new_status": keyword_status[item["id"]]["status"] if is_server_error else "failed_permanent"
                    })
            return chunk_results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_chunk, c) for c in chunks]
            for future in as_completed(futures):
                try:
                    for res in future.result(timeout=120):
                        g_id = res["global_id"]
                        if res["new_status"] in ["completed", "failed_permanent"]:
                            final_results[g_id] = res["data"]
                        keyword_status[g_id]["status"] = res["new_status"]
                except Exception: pass 
                completed_chunks += 1
                progress_bar.progress(min(completed_chunks / total_chunks, 1.0))
        pass_count += 1

    default_data = {"Intent": "error", "Funnel": "N/A"} if mode == "intent" else {"Topic": "error", "Subtopic": "error"}
    return [r if r is not None else default_data for r in final_results]

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Readme", "1. Intent Classifier", "2. Topic Mapper"])
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API Key. Speak to Tom if you don't have one.")

if page == "Readme":
    st.title("📖 How to Use This Tool")
    st.markdown("""
    ### Workflow Overview
    Follow these steps to classify your keywords accurately and efficiently:

    1.  **Gather Keywords:** Prepare a clean list of keywords. It is best to remove unnecessary data like search volumes or CPC at this stage to keep the file size manageable.
    2.  **API Keys:** Speak to Tom to obtain a valid Gemini API key.
    3.  **Step 1 - Intent Classification:**
        *   Navigate to **'1. Intent Classifier'**.
        *   Upload your keyword list as a `.csv` file.
        *   Select the column containing your keywords.
        *   Run the classifier and wait for the results.
    4.  **Export Data:** Download the resulting data as a `.csv` (e.g., `intent_results.csv`).
    5.  **Step 2 - Topic Mapping:**
        *   Navigate to **'2. Topic Mapper'**.
        *   Upload the exported `.csv` from Step 1.
    6.  **AI Suggestions:**
        *   Click **'Generate AI Suggestions'** to see recommended topics.
        *   Review these suggestions carefully.
        *   Manually add or remove topics in the sidebar based on your own validation and SEO strategy.
    7.  **Final Run:**
        *   Run the Topic Mapper.
        *   Export the final results for your report.
    """)
    st.info("💡 **Pro Tip:** Splitting the process into two steps ensures higher accuracy and prevents timeouts on large datasets.")

elif page == "1. Intent Classifier":
    st.title("Step 1: Intent & Funnel Classifier")
    st.info("Upload your raw keyword list to classify Search Intent and Marketing Funnel stage.")
    
    uploaded_file = st.file_uploader("Upload Keyword CSV", type=["csv"], key="intent_upload", 
                                     help="Upload a standard CSV file containing your keyword list.")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_col = st.selectbox("Keyword Column", df.columns, key="intent_col", 
                                  help="Select the column that contains the actual search queries.")
        
        if st.button("Run Intent Classification", help="Starts the AI classification for Search Intent and Funnel stage."):
            if not api_key:
                st.error("Missing API Key.")
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
                    st.download_button("📥 Download Intent Results", df.to_csv(index=False), "intent_results.csv", "text/csv", 
                                       help="Download the results to use in Step 2.")

else:
    st.title("Step 2: Custom Topic Mapper")
    st.info("Upload the CSV from Step 1 to add custom Topic and Subtopic categorisation.")
    
    with st.sidebar:
        st.markdown("---")
        st.write("### Classification Strategy")
        st.session_state.topics = st.text_area("Primary Topics", value=st.session_state.topics, height=150, 
                                               help="Enter primary topics here, one per line. These act as your 'buckets'.")
        st.session_state.subtopics = st.text_area("Subtopics (Optional)", value=st.session_state.subtopics, height=150, 
                                                  help="Enter subtopics here, one per line. These provide more granular mapping.")

    uploaded_file = st.file_uploader("Upload Intent CSV", type=["csv"], key="topic_upload", 
                                     help="Upload the file you exported from the Intent Classifier.")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_col = st.selectbox("Keyword Column", df.columns, key="topic_col", 
                                  help="Select the keyword column to map against your custom topics.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✨ Generate AI Suggestions", help="AI scans your keywords and suggests relevant topics/subtopics based on your niche."):
                if not api_key: st.error("Missing API Key.")
                else:
                    with st.spinner("Analysing sample..."):
                        unique_kws = [kw for kw in df[target_col].astype(str).unique() if kw.strip().lower() not in ['nan', 'null', '']]
                        sample = pd.Series(unique_kws).sample(n=min(80, len(unique_kws))).tolist()
                        st.session_state.ai_suggestions = suggest_topics(sample, api_key)
        with col2:
            if st.button("🗑️ Clear Suggestions", help="Clears the AI suggested text."):
                st.session_state.ai_suggestions = ""
                st.rerun()

        if st.session_state.ai_suggestions:
            st.code(st.session_state.ai_suggestions)
            st.caption("Copy these into the sidebar fields.")

        if st.button("Run Topic Mapping", help="Maps all keywords in your file to the topics and subtopics provided in the sidebar."):
            if not api_key: st.error("Missing API Key.")
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
                    st.download_button("📥 Download Final Results", df.to_csv(index=False), "final_seo_results.csv", "text/csv", 
                                       help="Download your fully classified keyword report.")
