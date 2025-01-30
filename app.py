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
GEMINI_API_KEY = 'AIzaSyAm_Fx8efZ2ELCwL0ZzZXMDMbrF6StdKsg'
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
    if pd.isna(description):
        return "• [No details available]"
    sentences = description.split('. ')
    takeaways = [f"• {sentence.strip()}" for sentence in sentences[:5] if sentence.strip()]
    while len(takeaways) < 5:
        takeaways.append("• [No further details available]")
    return '<br>'.join(takeaways)

def filter_data(df, business_area):
    if df is None:
        return pd.DataFrame()
    df_filtered = df[df["Business Area"] == business_area].copy()
    df_filtered["Key Takeaways"] = df_filtered["Description"].apply(generate_key_takeaways)
    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered[["Term Type", "Sub-Type", "Key Takeaways", "Page #"]]

def plot_pie_chart(data, show_labels=True):
    if data is None:
        return None
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
    try:
        with open(file_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

def process_pdf(uploaded_file):
    if uploaded_file is None:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        return text_splitter.split_text(text) if text.strip() else []
    except Exception as e:
        st.error(f"Failed to process PDF: {str(e)}")
        return []

def create_vector_store(texts):
    if not texts:
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_texts(texts=texts, embedding=embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store: {str(e)}")
        return None

def get_answer(question, vector_store, api_key):
    if not vector_store or not question.strip():
        return "I couldn't process your question. Please try again."
    
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
    
    # CSS styles
    st.markdown("""
        <style>
            .stRadio > div {
                background-color: transparent !important;
                padding: 0 !important;
            }
            .stRadio label {
                color: #ffffff !important;
                background-color: transparent !important;
            }
            .main-header {
                text-align: center;
                padding: 20px 0;
                margin-bottom: 30px;
            }
            .logo-title {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 20px;
            }
            .main-title {
                font-size: 48px;
                font-weight: bold;
                color: #FF5733;
            }
            .content-section {
                display: flex;
                gap: 20px;
                margin-top: 20px;
            }
            .chart-card {
                border: 2px solid #4a4a4a;
                border-radius: 10px;
                padding: 20px;
                background-color: #ffffff;
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
            .chat-message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 10px;
            }
            .user-message {
                background-color: #0078d4;
                color: white;
                margin-left: 20%;
            }
            .assistant-message {
                background-color: #4a4a4a;
                color: white;
                margin-right: 20%;
            }
            .business-area-section {
                background-color: transparent;
                padding: 20px;
                border-radius: 10px;
            }
            .stTextInput > div > div > input {
                background-color: #ffffff;
            }
            div[data-testid="stForm"] {
                border: none;
                padding: 0;
            }
        </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown("""
        <div class="main-header">
            <div class="logo-title">
                <span class="main-title">ContractIQ</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # File Uploader in sidebar
    uploaded_file = st.sidebar.file_uploader("Upload a contract file", type=["pdf"])

    # Initialize session state
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
        st.session_state.data = None
        st.session_state.vector_store = None
        st.session_state.messages = []

    # Process uploaded file
    if uploaded_file and st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.data = None
        
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

    # Main Content Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="business-area-section">', unsafe_allow_html=True)
        st.subheader("Select Business Area")
        business_area = st.radio(
            "",  # Empty label to remove default header
            ["Operational Risk Management", "Financial Risk Management"]
        )
        
        if st.button("Generate Report"):
            if st.session_state.data is not None:
                with st.spinner("Generating report..."):
                    time.sleep(2)
                    report = filter_data(st.session_state.data, business_area)
                    st.session_state.report = report
            else:
                st.warning("Please upload a document first.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if uploaded_file and st.session_state.data is not None:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.subheader("Business Area Distribution")
            fig = plot_pie_chart(st.session_state.data, show_labels=True)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Report Display
    if "report" in st.session_state and not st.session_state.report.empty:
        st.markdown("<div class='report-container'>", unsafe_allow_html=True)
        st.write(f"### Report for {business_area}")
        st.write(st.session_state.report.to_html(escape=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Chat Interface
    st.markdown("---")
    st.subheader("Document Chat Assistant")
    
    # Chat container with messages
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            class_name = "user-message" if message["role"] == "user" else "assistant-message"
            st.markdown(
                f'<div class="chat-message {class_name}">{message["content"]}</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input form
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
