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
    2. A list of deduplicated, concise SUBTOPIC 'stems' (up to 3 per topic).
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
    batch_size = 100 
    max_workers = 10 
    
    # Mapping for token reduction
    intent_map = {
        "1": "definition/factual", "2": "examples/list", "3": "comparison/pros-cons",
        "4": "asset/download/tool", "5": "product/service", "6": "instruction/how-to",
        "7": "consequence/effects/impacts", "8": "benefits/reason/justification",
        "9": "cost/price", "10": "unclear"
    }
    funnel_map = {"A": "Awareness", "C": "Consideration", "T": "Transactional"}

    system_instruction = f"""
    Strict SEO Analyst. Output: INDEX|INTENT_CODE|FUNNEL_CODE|CONFIDENCE|TOPIC|SUBTOPIC
    
    INTENT_CODES: {", ".join([f"{k}:{v}" for k,v in intent_map.items()])}
    FUNNEL_CODES: A:Awareness, C:Consideration, T:Transactional
    CONFIDENCE: 0.0-1.0
    TOPIC/SUBTOPIC: Use "N/A" if not EXPLICIT.
    """

    if custom_mode:
        # Compress lists to save input tokens
        c_topics = ",".join([t.strip() for t in topics.split("\n") if t.strip()])
        c_subtopics = ",".join([s.strip() for s in subtopics.split("\n") if s.strip()])
        system_instruction += f"\nTOPICS: {c_topics}\nSUBTOPICS: {c_subtopics}\nNo 'best fit'."

    final_results = [None] * len(keywords)
    keyword_status = [{"id": i, "kw": kw, "status": "pending"} for i, kw in enumerate(keywords)]
    
    pass_count = 1
    status_text = st.empty()
    progress_bar = st.progress(0)

    while any(s["status"] in ["pending", "retry_skip"] for s in keyword_status):
        to_process = [s for s in keyword_status if s["status"] in ["pending", "retry_skip"]]
        if not to_process or pass_count > 15:
            break
            
        if pass_count > 1:
            time.sleep(1)

        chunks = [to_process[i : i + batch_size] for i in range(0, len(to_process), batch_size)]
        total_chunks = len(chunks)
        completed_chunks = 0
        
        status_text.text(f"Pass {pass_count}: Processing {len(to_process)} keywords...")

        def process_chunk(chunk):
            mapping = {idx: item["id"] for idx, item in enumerate(chunk)}
            formatted = "\n".join([f"{idx}: {item['kw']}" for idx, item in enumerate(chunk)])
            
            chunk_results = []
            try:
                res = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.0,
                    ),
                    contents=f"Classify indices:\n{formatted}"
                )
                
                batch_data = {}
                for line in res.text.strip().split("\n"):
                    if "|" not in line: continue
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 3:
                        try:
                            idx = int(parts[0].strip("*: "))
                            # Map codes back to full words
                            intent = intent_map.get(parts[1], "unclear")
                            funnel = funnel_map.get(parts[2].upper(), "Awareness")
                            
                            batch_data[idx] = {
                                "Intent": intent, 
                                "Funnel": funnel,
                                "Confidence": float(parts[3]) if len(parts) > 3 and parts[3].replace(".","",1).isdigit() else 0.5,
                                "Topic": parts[4] if len(parts) > 4 else "N/A",
                                "Subtopic": parts[5] if len(parts) > 5 else "N/A"
                            }
                        except Exception: continue
                
                for idx in range(len(chunk)):
                    global_idx = mapping[idx]
                    item = batch_data.get(idx)
                    
                    if item:
                        chunk_results.append({"global_id": global_idx, "data": item, "new_status": "completed"})
                    else:
                        current_status = keyword_status[global_idx]["status"]
                        new_status = "failed_permanent" if current_status == "retry_skip" else "retry_skip"
                        chunk_results.append({
                            "global_id": global_idx,
                            "data": {"Intent": "skipped", "Funnel": "N/A", "Confidence": 0.0, "Topic": "N/A", "Subtopic": "N/A"},
                            "new_status": new_status
                        })
            except Exception as e:
                error_msg = str(e).lower()
                is_server_error = any(code in error_msg for code in ["503", "500", "504", "overloaded", "deadline", "timeout"])
                for item in chunk:
                    global_idx = item["id"]
                    chunk_results.append({
                        "global_id": global_idx,
                        "data": {"Intent": "error", "Funnel": "error", "Confidence": 0.0, "Topic": "error", "Subtopic": "error"},
                        "new_status": keyword_status[global_idx]["status"] if is_server_error else "failed_permanent"
                    })
            return chunk_results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_chunk, c) for c in chunks]
            for future in as_completed(futures):
                try:
                    batch_out = future.result(timeout=120) 
                    for res in batch_out:
                        g_id = res["global_id"]
                        if res["new_status"] in ["completed", "failed_permanent"]:
                            final_results[g_id] = res["data"]
                        keyword_status[g_id]["status"] = res["new_status"]
                except Exception: pass 
                completed_chunks += 1
                progress_bar.progress(min(completed_chunks / total_chunks, 1.0))
        pass_count += 1

    for i, res in enumerate(final_results):
        if res is None:
            final_results[i] = {"Intent": "error", "Funnel": "N/A", "Confidence": 0.0, "Topic": "N/A", "Subtopic": "N/A"}
    return final_results

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
