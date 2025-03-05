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
    if isinstance(description, list):
        return "<br>".join(description)
    return description

def filter_data(df, business_area):
    df_filtered = df[df["Business Area"] == business_area]
    df_filtered["Key Takeaways"] = df_filtered["Description"].apply(generate_key_takeaways)
    df_filtered.reset_index(drop=True, inplace=True)
    df_filtered.index = df_filtered.index + 1  # Start index from 1
    return df_filtered[["Term Type", "Sub-Type", "Key Takeaways", "Page #"]]

def plot_pie_chart(data):
    counts = data["Business Area"].value_counts()
    custom_colors = px.colors.sequential.RdBu
    if "Medicaid Compliance" in counts.index:
        custom_colors = list(custom_colors)
        medicaid_index = counts.index.get_loc("Medicaid Compliance")
        custom_colors[medicaid_index] = "#1f77b4"
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title="",
        color_discrete_sequence=custom_colors
    )
    fig.update_traces(
        textinfo="percent+label",
        textposition="inside",
        insidetextorientation='horizontal',
        pull=[0.1, 0],
        hole=0.2,
        textfont=dict(size=12)
    )
    fig.update_layout(
        height=400,
        width=600,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        uniformtext_minsize=10,
        uniformtext_mode='hide'
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
        model = genai.GenerativeModel('gemini-2.0-flash')
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
            .card {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                background: #f9f9f9;
            }
            .dropdown {
                margin-bottom: 20px;
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

        # Persona Cognition Model Card
        st.markdown("""
            <div class="card">
                <h3>Persona Cognition Model - IT Vendor Contracts</h3>
                <details>
                    <summary>View Summary</summary>
                    <table>
                        <tr><th>Commercial</th><th></th></tr>
                        <tr><td>Service Description</td><td>Cloud Services</td></tr>
                        <tr><td>Term of the Contract (Valid Till)</td><td>December 31, 2029</td></tr>
                        <tr><td>Contract Value</td><td>$3,000,000</td></tr>
                        <tr><td>Payment Terms</td><td>Net 30</td></tr>
                        <tr><th>Legal</th><th></th></tr>
                        <tr><td>Right to Terminate</td><td>Yes - With Cause</td></tr>
                        <tr><td>Right to Indemnify</td><td>Yes</td></tr>
                        <tr><td>Right to Assign</td><td>Yes - Assignable with Restrictions</td></tr>
                        <tr><td>Renewal Terms</td><td>Auto Renewal</td></tr>
                    </table>
                </details>
            </div>
        """, unsafe_allow_html=True)

        # Path - Document Upload Dropdown
        st.markdown("""
            <div class="dropdown">
                <details>
                    <summary><b>Path - Document Upload</b></summary>
                    <div style="margin-top: 10px;">
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a contract file", type=["pdf"], label_visibility="collapsed")
        st.markdown("</div></details></div>", unsafe_allow_html=True)

    # Rest of the code remains the same...
    # [Include the rest of your existing code here]

if __name__ == "__main__":
    main()
