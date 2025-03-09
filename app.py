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
                white-space: normal !important; /* Ensure text wraps */
                word-wrap: break-word; /* Break long words */
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
        persona_options = ["IT Vendor Contracts", "QWERTY", "POIUY"]
        selected_persona = st.selectbox(
            "Persona/ AI Model",
            options=persona_options,
            index=0,  # Default to "IT Vendor Contracts"
            key="persona_selectbox"
        )

        # Path - Dropdown
        path_options = ["Upload", "ASDF"]
        selected_path = st.selectbox(
            "Source",
            options=path_options,
            index=0,  # Default to "Upload"
            key="path_selectbox"
        )

        # File Uploader
        uploaded_file = st.file_uploader("Upload a contract file", type=["pdf"], label_visibility="collapsed")

        # Display the table only after a file is uploaded
        if uploaded_file:
            st.markdown("""
                <table class="summary-table">
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
            """, unsafe_allow_html=True)

    # Session State
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
        st.session_state.data = None
        st.session_state.business_area = None
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.session_state.selected_rows = []  # To store selected rows

    # Process Document
    if uploaded_file and st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.data = None
        st.session_state.business_area = None
        
        if "ACME" in uploaded_file.name.upper():
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
                ["Operational Risk", "Financial Risk", "Regulatory Risk", "Legal Risk"],
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
            # Create columns for title and button with proper vertical alignment
            col_title, col_btn = st.columns([4, 1])
            with col_title:
                st.markdown("<div class='section-title' style='margin-bottom: 0;'>Analysis Report</div>", unsafe_allow_html=True)
            with col_btn:
                st.markdown("<div style='margin-top: 28px;'>", unsafe_allow_html=True)  # Adjust margin to align button
                st.button("Export to Excel", key="export_btn")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Add a "Select" column with checkboxes
            report_with_checkbox = st.session_state.report.copy()
            report_with_checkbox.insert(0, "Select", False)

            # Configure column settings for text wrapping
            column_config = {
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select rows to send",
                    width="small"
                ),
                "Key Takeaways": st.column_config.TextColumn(
                    "Key Takeaways",
                    width="large"
                )
            }

            # Display the editable table
            edited_report = st.data_editor(
                report_with_checkbox,
                column_config=column_config,
                hide_index=True,
                use_container_width=True
            )

            # Add "Send to" dropdown and button
            email_options = ["abc@asd.com", "qwerr@wsde.com", "qswok@cvf.com"]
            selected_email = st.selectbox("Send to", options=email_options, key="email_selectbox")
            
            if st.button("Send to"):
                selected_rows = edited_report[edited_report["Select"]].index.tolist()
                if selected_rows:
                    st.success(f"Email sent to {selected_email} for selected rows: {selected_rows}")
                else:
                    st.warning("Please select at least one row to send.")

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
