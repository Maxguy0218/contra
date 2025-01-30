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

def plot_pie_chart(data):
    counts = data["Business Area"].value_counts()
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title="Business Area Distribution",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig.update_traces(textinfo="percent+label", pull=[0.1, 0], hole=0.2)
    fig.update_layout(height=500, width=500, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def process_pdf(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text.split("\n")
    except Exception as e:
        st.error(f"Failed to process PDF: {str(e)}")
        return []

def create_vector_store(texts):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_texts(texts=texts, embedding=embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store: {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide")

    # CSS Styling
    st.markdown("""
        <style>
            .centered-header {
                text-align: center;
            }
            .sidebar-title {
                font-size: 24px;
                font-weight: bold;
                color: #FF5733;
                text-align: center;
                padding-bottom: 20px;
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
        </style>
    """, unsafe_allow_html=True)

    # Branding
    logo_path = "logo.svg"
    logo_base64 = ""
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as img_file:
            logo_base64 = base64.b64encode(img_file.read()).decode()

    if logo_base64:
        st.markdown(f'<div class="centered-header"><img src="data:image/svg+xml;base64,{logo_base64}" width="100" /></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="centered-header">ContractIQ</h1>', unsafe_allow_html=True)

    # Layout: Sidebar + Main Content
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Select Business Area")
        business_area = st.radio("", ["Operational Risk Management", "Financial Risk Management"], label_visibility="collapsed")
        st.button("Generate Report")
    
    with col2:
        if "uploaded_file" in st.session_state:
            st.plotly_chart(plot_pie_chart(st.session_state.data), use_container_width=True)
    
    # Chat Section
    st.markdown("---")
    st.subheader("Document Chat Assistant")
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        role_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
        st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    with st.form(key="chat_form"):
        user_input = st.text_input("Ask about the contract:")
        submit_button = st.form_submit_button("Send")
    
    if submit_button and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": "[Placeholder response]"})
        st.rerun()

if __name__ == "__main__":
    main()
