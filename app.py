import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests
import json

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

# --- Logic: Ollama Helper ---
def call_ollama(prompt, system_instruction, model, url, response_schema=None):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    if response_schema:
        payload["format"] = "json"
        if "Intent" in str(response_schema):
            example = '{"results": [{"idx": 0, "i": "1", "f": "A"}, {"idx": 1, "i": "5", "f": "T"}]}'
        else:
            example = '{"results": [{"idx": 0, "t": "Topic", "s": "Subtopic"}, {"idx": 1, "t": "Topic", "s": "Subtopic"}]}'
        payload["messages"][0]["content"] += f" Return a JSON object with a 'results' key. YOU MUST INCLUDE THE 'idx' FOR EVERY ITEM. Example: {example}"

    try:
        session = requests.Session()
        session.trust_env = False 
        response = session.post(f"{url}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        
        print(f"--- OLLAMA DEBUG ({model}) ---")
        print(f"Content: {content[:200]}...")
        
        if response_schema:
            try:
                return response_schema.model_validate_json(content)
            except Exception:
                try:
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return response_schema.model_validate_json(json_match.group(0))
                except Exception: pass
                raise Exception(f"Failed to parse Ollama JSON: {content[:100]}...")
        return content
    except Exception as e:
        raise Exception(f"Ollama Error: {e}")

# --- Logic: Topic Suggester ---
def suggest_topics(sample_keywords, engine, config):
    if engine == "Gemini":
        system_instruction = "You are a technical SEO specialist. Provide a CONCISE list of topics. Output ONLY the requested blocks."
        limit_text = "Aim for a maximum of 5 primary TOPICS and up to 3 subtopics per topic."
    else:
        system_instruction = "You are a technical SEO specialist. Output ONLY the requested blocks. Do not include any conversational text, introductions, or formatting."
        limit_text = "Provide a comprehensive and diverse list of TOPICS and up to 5 subtopics per topic."

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
        if engine == "Gemini":
            client = genai.Client(api_key=config["api_key"])
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.0),
                contents=prompt
            )
            return response.text
        else:
            return call_ollama(prompt, system_instruction, config["model"], config["url"])
    except Exception as e:
        return f"Error: {e}"

# --- Logic: Generic Batch Processor ---
def process_batches(keywords, engine, config, mode, topics="", subtopics=""):
    model_id = "gemini-3-flash-preview"
    batch_size = 50 if engine == "Gemini" else 10
    max_workers = 5 if engine == "Gemini" else 1
    
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

    cache = None
    if engine == "Gemini":
        try:
            client = genai.Client(api_key=config["api_key"])
            cache = client.caches.create(
                model=model_id,
                config=types.CacheConfig(system_instruction=system_instruction, ttl_seconds=3600)
            )
        except Exception: pass

    final_results = [None] * len(keywords)
    # 0: pending, 1: completed, 2: failed_permanent, 3: retry_skip
    keyword_status = [0] * len(keywords)
    
    status_text = st.empty()
    progress_bar = st.progress(0.0)

    for pass_count in range(1, 11):
        to_process_indices = [i for i, status in enumerate(keyword_status) if status in [0, 3]]
        if not to_process_indices: break
        
        if pass_count > 1:
            time.sleep(min(pass_count * 2, 10))

        chunks = []
        for i in range(0, len(to_process_indices), batch_size):
            chunk_indices = to_process_indices[i : i + batch_size]
            chunks.append([(idx, keywords[idx]) for idx in chunk_indices])

        total_chunks = len(chunks)
        completed_chunks = 0
        status_text.text(f"Pass {pass_count}: Processing {len(to_process_indices)} keywords using {engine}...")

        def process_chunk(chunk):
            chunk_out = []
            try:
                formatted = "\n".join([f"{idx_in_batch}|{kw}" for idx_in_batch, (global_idx, kw) in enumerate(chunk)])
                mapping = {idx_in_batch: global_idx for idx_in_batch, (global_idx, kw) in enumerate(chunk)}
                
                if engine == "Gemini":
                    client = genai.Client(api_key=config["api_key"])
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
                    parsed = res.parsed
                else:
                    parsed = call_ollama(formatted, system_instruction, config["model"], config["url"], response_schema=schema)
                
                batch_data = {}
                if parsed and parsed.results:
                    for item in parsed.results:
                        if mode == "intent":
                            batch_data[item.idx] = {"Intent": intent_map.get(item.i, "unclear"), "Funnel": funnel_map.get(item.f.upper(), "Awareness")}
                        else:
                            batch_data[item.idx] = {"Topic": item.t, "Subtopic": item.s}
                
                for idx_in_batch in range(len(chunk)):
                    g_idx = mapping[idx_in_batch]
                    data = batch_data.get(idx_in_batch)
                    if data:
                        chunk_out.append((g_idx, data, 1)) # completed
                    else:
                        new_status = 2 if keyword_status[g_idx] == 3 else 3
                        chunk_out.append((g_idx, None, new_status))
            except Exception as e:
                error_msg = str(e).lower()
                is_server_error = any(code in error_msg for code in ["503", "500", "504", "overloaded", "deadline", "timeout", "quota", "429"])
                for global_idx, kw in chunk:
                    new_status = 3 if is_server_error else 2
                    chunk_out.append((global_idx, None, new_status))
            return chunk_out

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_chunk, c) for c in chunks]
            for future in as_completed(futures):
                try:
                    for g_idx, data, new_status in future.result(timeout=180):
                        if data is not None:
                            final_results[g_idx] = data
                        keyword_status[g_idx] = new_status
                except Exception: pass
                completed_chunks += 1
                progress_bar.progress(min(completed_chunks / total_chunks, 1.0))

    default_data = {"Intent": "error", "Funnel": "N/A"} if mode == "intent" else {"Topic": "error", "Subtopic": "error"}
    return [r if r is not None else default_data for r in final_results]

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Readme", "1. Intent Classifier", "2. Topic Mapper"])

st.sidebar.markdown("---")
st.sidebar.title("Engine Configuration")
engine = st.sidebar.radio("Select Engine", ["Gemini", "Ollama"], help="Choose between Cloud (Gemini) or Local (Ollama) processing.")

engine_config = {}
if engine == "Gemini":
    engine_config["api_key"] = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API Key. Speak to Tom if you don't have one.")
else:
    engine_config["url"] = st.sidebar.text_input("Ollama URL", value="http://127.0.0.1:11434", help="Local address of your Ollama server.")
    engine_config["model"] = st.sidebar.text_input("Ollama Model", value="llama3", help="The model name installed in Ollama (e.g., llama3, mistral, gemma).")

if page == "Readme":
    st.title("📖 How to Use This Tool")
    st.markdown("""
    ### Workflow Overview
    1.  **Gather Keywords:** Prepare a clean list of search queries in a `.csv` file.
    2.  **Select Engine:** In the sidebar, choose between **Gemini** (Cloud) or **Ollama** (Local).
        *   For Gemini: Speak to Tom for an API key.
        *   For Ollama: Ensure Ollama is running on your PC with the correct model installed.
    3.  **Step 1 - Intent Classification:**
        *   Go to **'1. Intent Classifier'**.
        *   Upload your CSV and select the keyword column.
        *   Run the classification and download `intent_results.csv`.
    4.  **Step 2 - Topic Mapping:**
        *   Go to **'2. Topic Mapper'**.
        *   Upload the file from Step 1.
        *   Generate AI Suggestions (optional) to build your strategy.
        *   Run Topic Mapping and export the final report.
    """)
    st.info("💡 **Pro Tip:** Splitting the process into two steps ensures higher accuracy and prevents timeouts on large datasets.")

elif page == "1. Intent Classifier":
    st.title("Step 1: Intent & Funnel Classifier")
    st.info(f"Using {engine} engine to classify Search Intent and Marketing Funnel stage.")
    
    uploaded_file = st.file_uploader("Upload Keyword CSV", type=["csv"], key="intent_upload")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_col = st.selectbox("Keyword Column", df.columns, key="intent_col")
        
        if st.button("Run Intent Classification"):
            if engine == "Gemini" and not engine_config.get("api_key"):
                st.error("Missing Gemini API Key.")
            else:
                with st.spinner(f"Classifying via {engine}..."):
                    raw_kws = df[target_col].astype(str).tolist()
                    unique_kws = list(dict.fromkeys([kw for kw in raw_kws if kw.strip() and kw.lower() != 'nan']))
                    results = process_batches(unique_kws, engine, engine_config, mode="intent")
                    
                    results_map = dict(zip(unique_kws, results))
                    final_results = [results_map.get(kw, {"Intent": "N/A", "Funnel": "N/A"}) for kw in raw_kws]
                    
                    res_df = pd.DataFrame(final_results)
                    df['Intent'], df['Funnel'] = res_df['Intent'], res_df['Funnel']
                    
                    st.success("Complete!")
                    st.dataframe(df)
                    st.download_button("📥 Download Intent Results", df.to_csv(index=False), "intent_results.csv", "text/csv")

else:
    st.title("Step 2: Custom Topic Mapper")
    st.info(f"Using {engine} engine to add custom Topic and Subtopic categorisation.")
    
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
                if engine == "Gemini" and not engine_config.get("api_key"): st.error("Missing Gemini API Key.")
                else:
                    with st.spinner("Analysing sample..."):
                        unique_kws = [kw for kw in df[target_col].astype(str).unique() if kw.strip().lower() not in ['nan', 'null', '']]
                        sample_size = 200 if engine == "Ollama" else 150
                        sample = pd.Series(unique_kws).sample(n=min(sample_size, len(unique_kws))).tolist()
                        st.session_state.ai_suggestions = suggest_topics(sample, engine, engine_config)
        with col2:
            if st.button("🗑️ Clear Suggestions"):
                st.session_state.ai_suggestions = ""
                st.rerun()

        if st.session_state.ai_suggestions:
            st.code(st.session_state.ai_suggestions)

        if st.button("Run Topic Mapping"):
            if engine == "Gemini" and not engine_config.get("api_key"): st.error("Missing Gemini API Key.")
            elif not st.session_state.topics: st.error("Provide topics in the sidebar.")
            else:
                with st.spinner(f"Mapping via {engine}..."):
                    raw_kws = df[target_col].astype(str).tolist()
                    unique_kws = list(dict.fromkeys([kw for kw in raw_kws if kw.strip() and kw.lower() != 'nan']))
                    results = process_batches(unique_kws, engine, engine_config, mode="topic", 
                                           topics=st.session_state.topics, subtopics=st.session_state.subtopics)
                    
                    results_map = dict(zip(unique_kws, results))
                    final_results = [results_map.get(kw, {"Topic": "N/A", "Subtopic": "N/A"}) for kw in raw_kws]
                    
                    res_df = pd.DataFrame(final_results)
                    df['Topic'], df['Subtopic'] = res_df['Topic'], res_df['Subtopic']
                    
                    st.success("Complete!")
                    st.dataframe(df)
                    st.download_button("📥 Download Final Results", df.to_csv(index=False), "final_seo_results.csv", "text/csv")
