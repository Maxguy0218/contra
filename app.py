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

# [Previous imports and configuration remain the same...]

def main():
    st.set_page_config(layout="wide")
    
    st.markdown("""
        <style>
            .stRadio > div {
                background-color: transparent !important;
                padding: 0 !important;
            }
            .stRadio label {
                color: #000000 !important;
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
            div[data-testid="stChatMessage"] {
                background-color: #ffffff;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }
            .chat-container {
                border: 2px solid #4a4a4a;
                border-radius: 10px;
                padding: 20px;
                height: 400px;
                overflow-y: auto;
                margin-top: 20px;
                background-color: #f5f5f5;
            }
            .user-message {
                background-color: #0078d4;
                color: white;
                padding: 10px;
                border-radius: 10px;
                margin: 5px 0;
                margin-left: 20%;
                max-width: 80%;
            }
            .assistant-message {
                background-color: #4a4a4a;
                color: white;
                padding: 10px;
                border-radius: 10px;
                margin: 5px 0;
                margin-right: 20%;
                max-width: 80%;
            }
            .business-area-section {
                background-color: transparent;
                padding: 20px;
                border-radius: 10px;
            }
            div[data-testid="stForm"] {
                border: none;
                padding: 0;
            }
        </style>
    """, unsafe_allow_html=True)

    # Get logo as base64
    logo_path = "logo.svg"
    logo_base64 = get_base64_image(logo_path) if os.path.exists(logo_path) else ""
    
    # Header with Logo and Title
    st.markdown(f"""
        <div class="main-header">
            <div class="logo-title">
                <img src="data:image/svg+xml;base64,{logo_base64}" style="width: 80px;" alt="Logo">
                <span class="main-title">ContractIQ</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # [File uploader and data processing remain the same...]

    # Initialize session state for chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Main Content Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="business-area-section">', unsafe_allow_html=True)
        st.subheader("Select Business Area")
        business_area = st.radio(
            "",
            ["Operational Risk Management", "Financial Risk Management"]
        )
        
        if st.button("Generate Report"):
            if st.session_state.get('data') is not None:
                with st.spinner("Generating report..."):
                    time.sleep(2)
                    report = filter_data(st.session_state.data, business_area)
                    st.session_state.report = report
            else:
                st.warning("Please upload a document first.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.get('data') is not None:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.subheader("Business Area Distribution")
            fig = plot_pie_chart(st.session_state.data, show_labels=True)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Display report if available
    if "report" in st.session_state and not st.session_state.report.empty:
        st.markdown("<div class='report-container'>", unsafe_allow_html=True)
        st.write(f"### Report for {business_area}")
        st.write(st.session_state.report.to_html(escape=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Chat Interface
    st.markdown("---")
    st.subheader("Document Chat Assistant")

    # Create a container for the chat messages
    chat_container = st.container()
    
    # Display chat messages in the container
    with chat_container:
        st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="user-message">{message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="assistant-message">{message["content"]}</div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask about the contract:", key="chat_input")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_input and st.session_state.get('vector_store'):
        st.session_state.messages.append({"role": "user", "content": user_input})
        try:
            answer = get_answer(user_input, st.session_state.vector_store, GEMINI_API_KEY)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Failed to generate answer: {str(e)}")
        st.rerun()

if __name__ == "__main__":
    main()
