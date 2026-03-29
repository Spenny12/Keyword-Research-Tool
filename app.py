import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    batch_size = 70  # Slightly reduced for better reliability during retries
    max_workers = 10
    
    system_instruction = """
    You are a strict, high-precision SEO analyst. For each keyword:
    1. Label intent: definition/factual, examples/list, comparison/pros-cons, asset/download/tool, product/service, instruction/how-to, consequence/effects/impacts, benefits/reason/justification, cost/price, unclear.
    2. Map to Marketing Funnel stage: Awareness, Consideration, or Transactional.
    3. Provide a confidence score (0.0 to 1.0).
    """

    if custom_mode:
        system_instruction += f"\nTOPICS LIST:\n{topics}\n\nSUBTOPICS LIST:\n{subtopics}\n"
        system_instruction += "\nCRITICAL RULES FOR TOPIC/SUBTOPIC:"
        system_instruction += "\n- ONLY assign a Topic or Subtopic if the keyword is EXPLICITLY and DIRECTLY related."
        system_instruction += "\n- Do NOT 'best fit' or guess. If the keyword is broad (e.g., 'composite decking') and the subtopic is specific (e.g., 'wood effect textures'), use 'N/A' for the subtopic."
        system_instruction += "\n- 'N/A' is the preferred answer if there is not a high-precision match."

    # Initialize final results container
    final_results = [None] * len(keywords)
    
    # Track which keywords need retrying
    # Status can be: "pending", "completed", "retry_skip", "failed_permanent"
    keyword_status = [{"id": i, "kw": kw, "status": "pending"} for i, kw in enumerate(keywords)]
    
    pass_count = 1
    status_text = st.empty()
    progress_bar = st.progress(0)

    while any(s["status"] in ["pending", "retry_skip"] for s in keyword_status):
        # Filter keywords that need processing in this pass
        to_process = [s for s in keyword_status if s["status"] in ["pending", "retry_skip"]]
        
        if not to_process:
            break
            
        chunks = [to_process[i : i + batch_size] for i in range(0, len(to_process), batch_size)]
        total_chunks = len(chunks)
        completed_chunks = 0
        
        status_text.text(f"Pass {pass_count}: Processing {len(to_process)} keywords in {total_chunks} batches...")

        def process_chunk(chunk):
            # Create a local mapping of index-in-chunk to global-index
            mapping = {idx: item["id"] for idx, item in enumerate(chunk)}
            formatted = "\n".join([f"{idx}: {item['kw']}" for idx, item in enumerate(chunk)])
            
            chunk_results = []
            try:
                res = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=BatchResponse,
                    ),
                    contents=f"Classify these keywords by index:\n{formatted}"
                )
                
                batch_data = {}
                if res.parsed and res.parsed.results:
                    batch_data = {item.index: item for item in res.parsed.results}
                
                for idx in range(len(chunk)):
                    global_idx = mapping[idx]
                    item = batch_data.get(idx)
                    
                    if item:
                        chunk_results.append({
                            "global_id": global_idx,
                            "data": {
                                "Intent": item.label, "Funnel": item.funnel_stage,
                                "Confidence": item.confidence, "Topic": item.topic, "Subtopic": item.subtopic
                            },
                            "new_status": "completed"
                        })
                    else:
                        # Logic for SKIPPED (missing from JSON)
                        # If this was already a retry for a skip, mark permanent
                        current_status = keyword_status[global_idx]["status"]
                        new_status = "failed_permanent" if current_status == "retry_skip" else "retry_skip"
                        
                        chunk_results.append({
                            "global_id": global_idx,
                            "data": {
                                "Intent": "skipped", "Funnel": "N/A", "Confidence": 0.0, "Topic": "N/A", "Subtopic": "N/A"
                            },
                            "new_status": new_status
                        })
            except Exception as e:
                error_msg = str(e)
                # Logic for 503 / Connection Errors (Always retry)
                is_server_error = any(code in error_msg for code in ["503", "500", "504"]) or "overloaded" in error_msg.lower() or "deadline" in error_msg.lower()
                
                for item in chunk:
                    global_idx = item["id"]
                    # If server error, keep as its CURRENT status (pending or retry_skip) so it tries again in the same capacity
                    chunk_results.append({
                        "global_id": global_idx,
                        "data": {"Intent": f"error: {error_msg[:30]}", "Funnel": "error", "Confidence": 0.0, "Topic": "error", "Subtopic": "error"},
                        "new_status": keyword_status[global_idx]["status"] if is_server_error else "failed_permanent"
                    })
            return chunk_results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_chunk, c) for c in chunks]
            for future in as_completed(futures):
                batch_out = future.result()
                for res in batch_out:
                    g_id = res["global_id"]
                    # Only update final_results if we actually got data or it's the final attempt
                    if res["new_status"] in ["completed", "failed_permanent"]:
                        final_results[g_id] = res["data"]
                    
                    keyword_status[g_id]["status"] = res["new_status"]
                
                completed_chunks += 1
                progress_bar.progress(min(completed_chunks / total_chunks, 1.0))
        
        pass_count += 1
        if pass_count > 15: # Safety break
            break

    # Fill any remaining gaps
    for i, res in enumerate(final_results):
        if res is None:
            final_results[i] = {"Intent": "error: max retries", "Funnel": "N/A", "Confidence": 0.0, "Topic": "N/A", "Subtopic": "N/A"}

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
