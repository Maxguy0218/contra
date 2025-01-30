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
        title="Business Area Distribution",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    textinfo = "percent+label" if show_labels else "none"
    fig.update_traces(textinfo=textinfo, pull=[0.1, 0], hole=0.2)
    fig.update_layout(height=400, width=600, 
                      margin=dict(l=20, r=20, t=40, b=20),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
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
                text-align: center;
                margin: -75px 0 -30px 0;
                padding: 0;
            }
            .main-title {
                font-size: 36px;
                font-weight: bold;
                color: #FF5733;
                display: inline-block;
                vertical-align: middle;
                margin: 0;
            }
            .logo-img {
                height: 50px;
                vertical-align: middle;
                margin-right: 15px;
            }
            .sidebar-logo {
                height: 40px;
                margin-right: 10px;
                vertical-align: middle;
            }
            .chart-card {
                border: 2px solid #4a4a4a;
                border-radius: 15px;
                padding: 20px;
                background: rgba(255, 255, 255, 0.1);
                margin: 20px 0;
            }
            .stRadio [role=radiogroup] {
                gap: 15px;
            }
            .stRadio [role=radio] {
                background: rgba(255, 255, 255, 0.1) !important;
                border: 1px solid #4a4a4a !important;
                padding: 15px !important;
                border-radius: 10px !important;
            }
            .chat-container {
                border: 2px solid #4a4a4a;
                border-radius: 10px;
                padding: 20px;
                height: 400px;
                overflow-y: auto;
                background: rgba(45, 52, 54, 0.8);
                margin-top: 20px;
            }
            .user-msg {
                color: #ffffff;
                padding: 12px 18px;
                margin: 10px 0;
                border-radius: 15px;
                background: #0078d4;
                max-width: 70%;
                margin-left: auto;
                word-break: break-word;
            }
            .assistant-msg {
                color: #ffffff;
                padding: 12px 18px;
                margin: 10px 0;
                border-radius: 15px;
                background: #4a4a4a;
                max-width: 70%;
                margin-right: auto;
                word-break: break-word;
            }
            .stButton button {
                background-color: #FF5733 !important;
                color: white !important;
                border-radius: 10px !important;
                padding: 12px 24px !important;
                font-size: 16px !important;
                transition: all 0.3s ease !important;
            }
            .stButton button:hover {
                background-color: #E54B2A !important;
                transform: scale(1.02);
            }
            .stMarkdown h3 {
                margin-top: -10px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar with Logo
    with st.sidebar:
        logo_base64 = get_base64_image("logo.svg") if os.path.exists("logo.svg") else ""
        st.markdown(f"""
            <div style="margin: -30px 0 20px 0;">
                <img src="data:image/svg+xml;base64,{logo_base64}" class="sidebar-logo">
                <span style="font-size: 24px; color: #FF5733; vertical-align: middle;">ContractIQ</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## Document Upload")
        uploaded_file = st.file_uploader("Upload a contract file", type=["pdf"], label_visibility="collapsed")

    # Main Header
    logo_base64 = get_base64_image("logo.svg") if os.path.exists("logo.svg") else ""
    st.markdown(f"""
        <div class="header-container">
            <h1 class="main-title">
                <img src="data:image/svg+xml;base64,{logo_base64}" class="logo-img">
                ContractIQ
            </h1>
        </div>
    """, unsafe_allow_html=True)

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
    if st.session_state.data is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Select Business Area")
            business_area = st.radio(
                "Select a Business Area",
                ["Operational Risk Management", "Financial Risk Management"],
                key="ba_radio"
            )
            
            if st.button("Generate Report", key="report_btn"):
                with st.spinner("Generating report..."):
                    time.sleep(1)
                    report = filter_data(st.session_state.data, business_area)
                    st.session_state.report = report

        with col2:
            with st.container():
                st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                show_labels = not st.checkbox("Hide Chart Labels", key="labels_check")
                st.plotly_chart(plot_pie_chart(st.session_state.data, show_labels=show_labels), 
                              use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # Report Display
        if "report" in st.session_state and not st.session_state.report.empty:
            st.markdown("### Analysis Report")
            st.write(st.session_state.report.to_html(escape=False), unsafe_allow_html=True)

    # Chat Interface
    st.markdown("---")
    st.markdown("### Document Chat Assistant")
    
    with st.container():
        chat_col, _ = st.columns([3, 1])
        with chat_col:
            chat_container = st.container()
            with chat_container:
                st.markdown("<div class='chat-container' id='chat-box'>", unsafe_allow_html=True)
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f"<div class='user-msg'>{msg['content']}</div>", 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='assistant-msg'>{msg['content']}</div>", 
                                  unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with st.form(key="chat_form"):
                user_input = st.text_input("Ask about the contract:", 
                                         key="chat_input",
                                         label_visibility="collapsed")
                submit_button = st.form_submit_button("Send Message")

    if submit_button and user_input and st.session_state.vector_store:
        st.session_state.messages.append({"role": "user", "content": user_input})
        try:
            answer = get_answer(user_input, st.session_state.vector_store, GEMINI_API_KEY)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Failed to generate answer: {str(e)}")
        st.rerun()

    # Scroll to bottom of chat
    st.markdown("""
        <script>
            window.addEventListener('load', function() {
                var chatBox = document.getElementById('chat-box');
                chatBox.scrollTop = chatBox.scrollHeight;
            });
        </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
