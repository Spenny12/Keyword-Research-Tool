import streamlit as st
import pandas as pd
from openai import OpenAI

# --- UI Configuration ---
st.set_page_config(page_title="SEO Intent Classifier", layout="wide")
st.title("🎯 Keyword Content Format Classifier")
st.write("Upload a CSV/XLSX to categorize keyword intent for content strategy.")

# --- Sidebar: API Configuration ---
with st.sidebar:
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    model_choice = st.selectbox("Model", ["gpt-4o", "gpt-3.5-turbo"])
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.1)

# --- Logic: Classification Function ---
def classify_keywords(keywords, api_key, model):
    client = OpenAI(api_key=api_key)
    results = []
    
    # Updated Prompt for better LLM performance
    system_prompt = """
    You are an SEO expert. Label the provided keyword by the implied content format 
    ideal for the intent/endpoint of the query. 
    
    Return ONLY the lowercase label from this specific list:
    - definition/factual
    - examples/list
    - comparison/pros-cons
    - asset/download/tool
    - product/service
    - instruction/how-to
    - consequence/effects/impacts
    - benefits/reason/justification
    - cost/price
    
    If the intent is not explicit or easily intuited, return 'unclear'.
    """

    progress_bar = st.progress(0)
    
    for i, kw in enumerate(keywords):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Keyword: {kw}"}
            ],
            temperature=temperature
        )
        results.append(response.choices[0].message.content.strip().lower())
        progress_bar.progress((i + 1) / len(keywords))
        
    return results

# --- Main App Interface ---
uploaded_file = st.file_誠("Upload Keyword List", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    # Column selection
    target_col = st.selectbox("Select the column containing keywords:", df.columns)
    
    if st.button("Run Classification"):
        if not api_key:
            st.error("Please add your API key in the sidebar.")
        else:
            with st.spinner("Classifying..."):
                keywords = df[target_col].tolist()
                labels = classify_keywords(keywords, api_key, model_choice)
                
                df['Intent Label'] = labels
                
                st.success("Classification Complete!")
                st.dataframe(df)
                
                # Download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Classified CSV", csv, "classified_keywords.csv", "text/csv")
