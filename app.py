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
    custom_colors = px.colors.qualitative.Pastel  # Softer and more pleasing colors
    if "Regulatory Risk" in counts.index:
        custom_colors = list(custom_colors)
        regulatory_index = counts.index.get_loc("Regulatory Risk")
        custom_colors[regulatory_index] = "#A6CEE3"  # Softer blue for Regulatory Risk
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
        textfont=dict(size=14, color='black', family='Arial', weight='bold')  # Darker and bolder text
    )
    fig.update_layout(
        height=400,
        width=600,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        uniformtext_minsize=12,
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

# Define the data for the tables
def get_critical_data_insights(num_files):
    data = {
        "Engagement": ["IT Services", "IT - Services", "IT - Services", "IT - Services", 
                      "IT - Services", "IT - Services", "IT - Services", "IT - Services",
                      "IT - Services", "IT-Infrastructure", "IT-Infrastructure",
                      "IT-Infrastructure", "IT-Infrastructure"],
        "Type of Contract": ["SOW", "SOW", "EULA", "EUSA", "SOW", "SOW", "SOW", 
                            "EULA", "EUSA", "SOW", "EUSA", "EUSA", "EULA"],
        "Contract Coverage": ["FedEx Enterprise", "FedEx", "FedEx Express", "FedEx Ground",
                            "FedEx Freight", "FedEx Services", "FedEx", "FedEx Freight",
                            "FedEx", "FedEx Enterprise", "FedEx Enterprise",
                            "FedEx", "FedEx Enterprise"],
        "Geographical Scope": ["US", "US", "US, EMEA", "US", "US, Canada", "US", "APAC",
                             "US, LATAM", "Global", "US", "US, EMEA", "US, EMEA, APAC", "Global"],
        "Contract Scope": ["Professional Services AI", "Professional Services (AI & Analytics)",
                         "Data Management Software Licenses", "Networking Infrastructure & Support",
                         "Hybrid Cloud Deployment & Support", "Storage Solutions & Managed Services",
                         "IT Support & Maintenance Services", "Software Licenses (Compliance & Security)",
                         "Global Software Agreement & Support", "Servers Procurement & Maintenance",
                         "Cybersecurity Services (Vulnerability Mgmt)",
                         "Managed Detection and Response (MDR)", "Microsoft Azure Cloud Services"],
        "Effective From": ["12/1/2024", "1/7/2024", "12/31/2022", "7/31/2024", "1/31/2025",
                         "12/31/2023", "3/31/2024", "2/23/2025", "3/31/2025", "11/30/2024",
                         "12/15/2024", "12/20/2024", "7/14/2024"],
        "Expiry Date": ["3/31/2026", "6/30/2027", "12/31/2024", "4/30/2026", "8/31/2028",
                       "9/20/2025", "5/20/2025", "4/12/2026", "8/31/2029", "6/30/2027",
                       "11/30/2028", "10/31/2026", "1/15/2027"],
        "Status": ["Active", "Active", "Expired", "Active", "Active", "Active", "Active",
                  "Active", "Active", "Active", "Active", "Active", "Active"],
        "Auto-renewal": ["No", "", "Yes", "", "Yes", "No", "Yes", "Yes", "No", "", "No", "Yes", "Yes"]
    }
    return pd.DataFrame({k: v[:num_files] for k, v in data.items()})

def get_commercial_insights(num_files):
    data = {
        "Total Contract Value": ["$5,250,785", "$6,953,977", "$2,400,000", "$4,750,000",
                               "$6,200,202", "$1,850,000", "$3,309,753", "$1,275,050",
                               "$7,500,060", "$4,409,850", "$2,750,075", "$3,950,040",
                               "$8,250,070"],
        "Payment Terms": ["Net 60", "Net 60", "Net 45", "Net 60", "Net 60", "Net 45",
                        "Net 45", "Net 30", "Net 60", "Net 60", "Net 45", "Net 45", "Net 60"],
        "Early Payment Discount %": ["NIL", "NIL", "1.25% within 20 days", "1% within 10 days",
                                   "", "NIL", "0.2% within 15 days", "", "0.2% within 15 days",
                                   "NIL", "1.5% within 15 days", "", "0.625% within 15 days"],
        "Late Payment Penalty%": ["1.50%", "1.50%", "1%", "1.50%", "", "1%", "1%", "",
                                 "1.50%", "", "1.25%", "", "1.50%"],
        "Volume based Discounts %": ["5% for spend > $3M", "5% for spend > $3M", "",
                                   "8% for equipment > $1M", "7% for spend > $5M",
                                   "4% for equipment > $5 M", "", "5% over 15000 licenses",
                                   "10% for spend > $6M", "8% for spend > $2M",
                                   "6% for annual spend > $2M", "", "10% for spend > $7M"],
        "Annual Price Increase %": ["CPI + 1.5%", "CPI + 1.5%", "", "CPI", "CPI", "",
                                  "CPI -1%", "CPI", "CPI + .6%", "CPI + 0.25%", "CPI", "", "CPI + 1%"]
    }
    return pd.DataFrame({k: v[:num_files] for k, v in data.items()})

def get_legal_insights(num_files):
    data = {
        "Right to Indemnify": ["Yes"] * 13,
        "Right to Assign": ["No"] * 13,
        "Right to Terminate": ["Yes"] * 13,
        "Governing Law": ["Tennessee", "Tennessee", "UK", "Tennessee", "Tennessee",
                         "Tennessee", "Singapore", "Delaware", "Tennessee", "UK",
                         "Delaware", "Singapore", "Delaware"],
        "Liability Limit": ["TCV/ Higher or unl fdir IP/Conf", "", "", "", "", "", "", "", "", "", "", "", ""]
    }
    return pd.DataFrame({k: v[:num_files] for k, v in data.items()})

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
            .summary-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
                background-color: #2d3436; /* Dark background */
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .summary-table th, .summary-table td {
                border: 1px solid #444;
                padding: 12px;
                text-align: left;
                color: #ffffff; /* Light font for contrast */
            }
            .summary-table th {
                background-color: #4a4a4a; /* Slightly lighter header */
                color: #ffffff;
                font-weight: bold;
            }
            .summary-table tr:nth-child(even) {
                background-color: #3a3a3a; /* Alternate row color */
            }
            .summary-table tr:hover {
                background-color: #555; /* Hover effect */
            }
            .dropdown {
                margin-bottom: 20px;
            }
            .dropdown-arrow {
                margin-right: 10px;
                font-size: 18px;
            }
            .attribute-response {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 10px;
            }
            .attribute {
                font-weight: bold;
                color: #FFA07A; /* Softer color for Persona/ AI Model */
            }
            .response {
                color: #D3D3D3; /* Light green for IT Vendor Contracts */
            }
            .path-attribute {
                font-weight: bold;
                color: #FFA07A; /* Softer color for Path */
            }
            /* Tab styling */
            .stTabs [role=tablist] {
                gap: 10px;
                margin-bottom: 20px;
            }
            .stTabs [role=tab] {
                padding: 10px 20px;
                border-radius: 10px 10px 0 0;
                background-color: #3a3a3a;
                color: white;
                font-weight: bold;
                border: none;
            }
            .stTabs [role=tab][aria-selected=true] {
                background-color: #FF4500;
                color: white;
            }
            .stTabs [role=tab]:hover {
                background-color: #555;
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

        # Persona Cognition Model - Dropdown
        persona_options = ["IT Vendor Services", "IT Vendor Infra", "Commercial"]
        selected_persona = st.selectbox(
            "Persona/ AI Model",
            options=persona_options,
            index=0,  # Default to "IT Vendor Contracts"
            key="persona_selectbox"
        )

        # Path - Dropdown
        path_options = ["Local Machine", "Network Path"]
        selected_path = st.selectbox(
            "Source",
            options=path_options,
            index=0,  # Default to "Local Machine"
            key="path_selectbox"
        )

        # File Uploader - now supports multiple files
        uploaded_files = st.file_uploader("Upload contract files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    # Session State
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = None
        st.session_state.data = None
        st.session_state.business_area = None
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.session_state.active_tab = "Critical Data Insights"

    # Process Document
    if uploaded_files and st.session_state.uploaded_files != uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.data = None
        st.session_state.business_area = None
        st.session_state.active_tab = "Critical Data Insights"
        
        # For demo purposes, we'll just count the number of files uploaded
        num_files = len(uploaded_files)
        if num_files > 13:
            num_files = 13  # Cap at 13 records as per your example data
        
        # Store the number of files for table display
        st.session_state.num_files = num_files
        
        # Process all files (in a real app, you'd process each file)
        with st.spinner("Processing documents..."):
            all_texts = []
            for uploaded_file in uploaded_files:
                texts = process_pdf(uploaded_file)
                if texts:
                    all_texts.extend(texts)
            
            if all_texts:
                st.session_state.vector_store = create_vector_store(all_texts)
                st.session_state.messages = []

    # Main Content
    if st.session_state.uploaded_files:
        # Create tabs
        tabs = st.tabs(["Critical Data Insights", "Commercial Insights", "Legal Insights"])
        
        with tabs[0]:
            st.session_state.active_tab = "Critical Data Insights"
            critical_data = get_critical_data_insights(st.session_state.num_files)
            st.dataframe(critical_data, use_container_width=True)
        
        with tabs[1]:
            st.session_state.active_tab = "Commercial Insights"
            commercial_data = get_commercial_insights(st.session_state.num_files)
            st.dataframe(commercial_data, use_container_width=True)
        
        with tabs[2]:
            st.session_state.active_tab = "Legal Insights"
            legal_data = get_legal_insights(st.session_state.num_files)
            st.dataframe(legal_data, use_container_width=True)

        # Chat Interface (unchanged from original)
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
