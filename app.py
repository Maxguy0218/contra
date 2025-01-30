import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.express as px
import base64
import pdfplumber
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Hardcoded API Key (Replace with your actual key)
GEMINI_API_KEY = "AIzaSyAm_Fx8efZ2ELCwL0ZzZXMDMbrF6StdKsg"

# Path configurations
ATENA_JSON_PATH = os.path.join(os.getcwd(), "atena_annotations_fixed.json")
BCBS_JSON_PATH = os.path.join(os.getcwd(), "bcbs_annotations_fixed.json")

def load_atena_data():
    with open(ATENA_JSON_PATH, "r") as f:
        return pd.DataFrame(json.load(f))

def load_bcbs_data():
    with open(BCBS_JSON_PATH, "r") as f:
        return pd.DataFrame(json.load(f))

def generate_key_takeaways(description):
    sentences = description.split('. ')
    takeaways = [f"• {sentence.strip()}" for sentence in sentences[:5]]
    while len(takeaways) < 5:
        takeaways.append("• [No further details available]")
    return '<br>'.join(takeaways)

def filter_data(df, business_area):
    df_filtered = df[df["Business Area"] == business_area]
    df_filtered["Key Takeaways"] = df_filtered["Description"].apply(generate_key_takeaways)
    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered[["Term Type", "Sub-Type", "Key Takeaways", "Page #"]]

def plot_pie_chart(data, show_labels=True):
    counts = data["Business Area"].value_counts()
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title="",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    textinfo = "percent+label" if show_labels else "none"
    fig.update_traces(textinfo=textinfo, pull=[0.1, 0], hole=0.2)
    fig.update_layout(height=500, width=700, margin=dict(l=20, r=20, t=20, b=20))
    return fig

def get_base64_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def process_pdf(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Failed to process PDF: {str(e)}")
        return []

def create_vector_store(texts):
    if not texts:
        st.error("No text extracted from document")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_texts(texts=texts, embedding=embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store: {str(e)}")
        return None

def get_answer(question, vector_store, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        docs = vector_store.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        response = model.generate_content(f"Answer based on this context only:\n{context}\n\nQuestion: {question}")
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def main():
    st.set_page_config(layout="wide")
    
    st.markdown("""
        <style>
            .header-container {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: -60px;
                padding-top: 0;
            }
            .main-title {
                font-size: 48px;
                font-weight: bold;
                color: #FF5733;
                margin-top: -30px;
            }
            .sidebar-title {
                font-size: 24px;
                font-weight: bold;
                color: #FF5733;
                display: flex;
                align-items: center;
                gap: 10px;
                margin-top: -20px;
            }
            .report-container {
                max-width: 100%;
                margin: auto;
            }
            .chat-container {
                border: 2px solid #4a4a4a;
                border-radius: 10px;
                padding: 20px;
                height: 400px;
                overflow-y: auto;
                background: #2d3436;
                margin-top: 20px;
            }
            .user-msg {
                color: #ffffff;
                padding: 10px;
                margin: 5px 0;
                border-radius: 15px;
                background: #0078d4;
                max-width: 80%;
                margin-left: auto;
            }
            .assistant-msg {
                color: #ffffff;
                padding: 10px;
                margin: 5px 0;
                border-radius: 15px;
                background: #4a4a4a;
                max-width: 80%;
                margin-right: auto;
            }
            .chart-container {
                border: 2px solid #4a4a4a;
                border-radius: 10px;
                padding: 20px;
                margin-top: -20px;
            }
            .chart-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: -40px;
                padding: 10px;
                border-bottom: 1px solid #4a4a4a;
            }
            .stRadio > div {
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Branding
    logo_path = "logo.svg"
    logo_base64 = ""
    if os.path.exists(logo_path):
        logo_base64 = get_base64_image(logo_path)
    logo_img = f'<img src="data:image/svg+xml;base64,{logo_base64}" style="width: 50px;">' if logo_base64 else ""

    st.sidebar.markdown(f"""
        <div class="sidebar-title">
            {logo_img}
            ContractIQ
        </div>
    """, unsafe_allow_html=True)

    # Main Header
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f"""
            <div class="header-container">
                <img src="data:image/svg+xml;base64,{logo_base64}" style="width: 120px;">
                <span class="main-title">ContractIQ</span>
            </div>
        """, unsafe_allow_html=True)

    # File Uploader
    uploaded_file = st.sidebar.file_uploader("Upload a contract file", type=["pdf"])

    # Session State
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
        st.session_state.data = None
        st.session_state.business_area = None
        st.session_state.vector_store = None
        st.session_state.messages = []

    # Process Document
    if uploaded_file and st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.data = None
        st.session_state.business_area = None
        
        if "AETNA" in uploaded_file.name.upper():
            st.session_state.data = load_atena_data()
        elif "BLUE" in uploaded_file.name.upper():
            st.session_state.data = load_bcbs_data()
        else:
            st.error("ERROR: Unsupported document type.")
            return
        
        with st.spinner("Processing document..."):
            texts = process_pdf(uploaded_file)
            st.session_state.vector_store = create_vector_store(texts)
            st.session_state.messages = []

    # Main Content
    col1, col2 = st.columns([2, 4])
    
    with col1:
        st.subheader("Select Business Area")
        business_area = st.radio(
            "Select a Business Area",
            ["Operational Risk Management", "Financial Risk Management"]
        )
        
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                time.sleep(2)
                report = filter_data(st.session_state.data, business_area)
                st.session_state.report = report
    
    with col2:
        if uploaded_file and st.session_state.data is not None:
            show_labels = not st.sidebar.expander("Options").checkbox("Hide Labels")
            with st.container():
                st.markdown("""
                    <div class="chart-container">
                        <div class="chart-header">
                            <h3>Business Area Distribution</h3>
                        </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(plot_pie_chart(st.session_state.data, show_labels=show_labels), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Report Display
    if "report" in st.session_state and not st.session_state.report.empty:
        st.markdown("<div class='report-container'>", unsafe_allow_html=True)
        st.write(f"### Report for {business_area}")
        st.write(st.session_state.report.to_html(escape=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Chat Interface
    st.markdown("---")
    st.subheader("Document Chat Assistant")
    
    with st.markdown("<div class='chat-container'>", unsafe_allow_html=True):
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-msg'>{msg['content']}</div>", unsafe_allow_html=True)

        with st.form(key="chat_form"):
            user_input = st.text_input("Ask about the contract:", key="chat_input")
            submit_button = st.form_submit_button("Send")
    
    if submit_button and user_input and st.session_state.vector_store:
        st.session_state.messages.append({"role": "user", "content": user_input})
        try:
            answer = get_answer(user_input, st.session_state.vector_store, GEMINI_API_KEY)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Failed to generate answer: {str(e)}")
        st.rerun()

if __name__ == "__main__":
    main()
