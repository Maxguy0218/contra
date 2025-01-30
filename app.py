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
GEMINI_API_KEY ='AIzaSyAm_Fx8efZ2ELCwL0ZzZXMDMbrF6StdKsg'
# Path configurations remain the same
ATENA_JSON_PATH = os.path.join(os.getcwd(), "atena_annotations_fixed.json")
BCBS_JSON_PATH = os.path.join(os.getcwd(), "bcbs_annotations_fixed.json")

# Your existing helper functions remain the same
def load_atena_data():
    with open(ATENA_JSON_PATH, "r") as f:
        return pd.DataFrame(json.load(f))

def load_bcbs_data():
    with open(BCBS_JSON_PATH, "r") as f:
        return pd.DataFrame(json.load(f))

# Other helper functions remain the same...
# (generate_key_takeaways, filter_data, plot_pie_chart, get_base64_image, process_pdf, create_vector_store, get_answer)

def main():
    st.set_page_config(layout="wide")
    
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
                overflow-y: scroll;
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
        </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown("""
        <div class="main-header">
            <div class="logo-title">
                <img src="logo.svg" style="width: 80px;">
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
            with st.spinner("Generating report..."):
                time.sleep(2)
                report = filter_data(st.session_state.data, business_area)
                st.session_state.report = report
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if uploaded_file and st.session_state.data is not None:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.subheader("Business Area Distribution")
            st.plotly_chart(plot_pie_chart(st.session_state.data, show_labels=True), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

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
