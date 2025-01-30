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

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(layout="wide")

# Hardcoded API Key (Replace with your actual key)
GEMINI_API_KEY = 'AIzaSyAm_Fx8efZ2ELCwL0ZzZXMDMbrF6StdKsg'
# Path configurations
ATENA_JSON_PATH = os.path.join(os.getcwd(), "atena_annotations_fixed.json")
BCBS_JSON_PATH = os.path.join(os.getcwd(), "bcbs_annotations_fixed.json")

# Add JavaScript detection for screen width
st.markdown("""
    <script>
    window.addEventListener('load', function() {
        const width = window.innerWidth;
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: width
        }, '*');
    });
    </script>
    """, unsafe_allow_html=True)

def load_atena_data():
    with open(ATENA_JSON_PATH, "r") as f:
        return pd.DataFrame(json.load(f))

def load_bcbs_data():
    with open(BCBS_JSON_PATH, "r") as f:
        return pd.DataFrame(json.load(f))

def generate_key_takeaways(description):
    if isinstance(description, list):
        return "<br>".join(description)
    return description

def filter_data(df, business_area):
    df_filtered = df[df["Business Area"] == business_area]
    df_filtered["Key Takeaways"] = df_filtered["Description"].apply(generate_key_takeaways)
    df_filtered.reset_index(drop=True, inplace=True)
    df_filtered.index = df_filtered.index + 1
    return df_filtered[["Term Type", "Sub-Type", "Key Takeaways", "Page #"]]

def plot_pie_chart(data):
    counts = data["Business Area"].value_counts()
    
    custom_colors = px.colors.sequential.RdBu
    if "Medicaid Compliance" in counts.index:
        custom_colors = list(custom_colors)
        medicaid_index = counts.index.get_loc("Medicaid Compliance")
        custom_colors[medicaid_index] = "#1f77b4"

    # Detect screen width from session state
    screen_width = st.session_state.get('screen_width', 1200)
    
    # Adjust label positions based on screen width
    if screen_width < 1200:  # Sidebar likely visible
        textposition = 'outside'
        textangle = 0
        insidetextorientation = None
        textfont_size = 12
        legend_x = 1.2
    else:  # Full width mode
        textposition = 'inside'
        textangle = 0
        insidetextorientation = 'horizontal'
        textfont_size = 14
        legend_x = 1
    
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title="",
        color_discrete_sequence=custom_colors
    )
    
    fig.update_traces(
        textinfo="percent+label",
        textposition=textposition,
        insidetextorientation=insidetextorientation,
        textangle=textangle,
        pull=[0.1, 0],
        hole=0.2,
        textfont=dict(size=textfont_size),
        marker=dict(line=dict(color='#ffffff', width=2))
    )
    
    fig.update_layout(
        height=400,
        width=600,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        uniformtext_minsize=10,
        uniformtext_mode='hide',
        legend=dict(x=legend_x, y=0.5)
    )
    
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
    # Get screen width from frontend
    if 'screen_width' not in st.session_state:
        st.session_state.screen_width = 1200
    
    # Update screen width from JS
    if 'component_value' in st.session_state:
        st.session_state.screen_width = st.session_state.component_value

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
                color: #FF4500;
                display: inline-block;
                vertical-align: middle;
                margin: 0;
            }
            .logo-img {
                height: 50px;
                vertical-align: middle;
                margin-right: 15px;
            }
            .sidebar-title {
                font-size: 24px;
                color: #FF4500;
                font-weight: bold;
                margin: -10px 0 20px 0;
            }
            .section-title {
                font-size: 24px;
                font-weight: bold;
                color: #FF4500;
                margin: 20px 0;
            }
            .stRadio [role=radiogroup] {
                gap: 15px;
            }
            .stRadio [role=radio] {
                padding: 15px !important;
                border-radius: 10px !important;
            }
            .user-msg {
                color: #ffffff;
                padding: 12px 18px;
                margin: 10px 0;
                border-radius: 15px;
                background: #0078d4;
                word-break: break-word;
            }
            .assistant-msg {
                color: #ffffff;
                padding: 12px 18px;
                margin: 10px 0;
                border-radius: 15px;
                background: #4a4a4a;
                word-break: break-word;
            }
            .dataframe thead th {
                text-align: left !important;
            }
            .dataframe td:first-child {
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

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

    # Sidebar
    with st.sidebar:
        st.markdown(f"""
            <div class="sidebar-title">
                <img src="data:image/svg+xml;base64,{logo_base64}" style="height: 40px; vertical-align: middle;">
                ContractIQ
            </div>
        """, unsafe_allow_html=True)
        st.markdown("## Document Upload")
        uploaded_file = st.file_uploader("Upload a contract file", type=["pdf"], label_visibility="collapsed")

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
        # Section Titles
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-title'>Select Business Area</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='section-title'>Business Area Distribution</div>", unsafe_allow_html=True)

        # Content Columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            business_area = st.radio(
                "Select a Business Area",
                ["Operational Risk Management", "Financial Risk Management", "Medicaid Compliance"],
                key="ba_radio",
                label_visibility="collapsed"
            )
            
            if st.button("Generate Report", key="report_btn"):
                with st.spinner("Generating report..."):
                    time.sleep(1)
                    report = filter_data(st.session_state.data, business_area)
                    st.session_state.report = report

        with col2:
            st.plotly_chart(plot_pie_chart(st.session_state.data), use_container_width=True)

        # Divider line
        st.markdown("---")

        # Report Display
        if "report" in st.session_state and not st.session_state.report.empty:
            st.markdown("<div class='section-title'>Analysis Report</div>", unsafe_allow_html=True)
            st.write(st.session_state.report.to_html(escape=False), unsafe_allow_html=True)

        # Chat Interface
        st.markdown("<div class='section-title'>Document Assistant</div>", unsafe_allow_html=True)
        with st.container():
            with st.form(key="chat_form"):
                user_input = st.text_input("Ask about the contract:", key="chat_input")
                submit_button = st.form_submit_button("Ask Question")

        if submit_button and user_input and st.session_state.vector_store:
            try:
                answer = get_answer(user_input, st.session_state.vector_store, GEMINI_API_KEY)
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Failed to generate answer: {str(e)}")

        # Display chat history
        if st.session_state.messages:
            with st.container():
                st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "user":
                        st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='assistant-msg'>{msg['content']}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
