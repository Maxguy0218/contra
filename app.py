import streamlit as st
import pandas as pd
import plotly.express as px
import os
import base64
import pdfplumber
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Configuration
GEMINI_API_KEY = 'AIzaSyAm_Fx8efZ2ELCwL0ZzZXMDMbrF6StdKsg'
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# FedEx Colors
PRIMARY_COLOR = "#4D148C"  # Purple
SECONDARY_COLOR = "#FF6200"  # Orange
BACKGROUND_COLOR = "#FFFFFF"  # White
TEXT_COLOR = "#333333"  # Dark Gray

# Sample Data - Reduced to 5 sample contracts for demo
SAMPLE_CONTRACTS = [
    {
        "Contract Name": "FedEx Express Master Agreement",
        "Type": "Master",
        "Effective Date": "2023-01-15",
        "Expiration Date": "2026-01-14",
        "Value": "$4,200,000",
        "Status": "Active"
    },
    {
        "Contract Name": "FedEx Ground Service Agreement",
        "Type": "Service",
        "Effective Date": "2022-06-01",
        "Expiration Date": "2025-05-31",
        "Value": "$3,750,000",
        "Status": "Active"
    },
    {
        "Contract Name": "FedEx Freight Transportation Contract",
        "Type": "Transportation",
        "Effective Date": "2023-03-10",
        "Expiration Date": "2024-03-09",
        "Value": "$2,100,000",
        "Status": "Active"
    },
    {
        "Contract Name": "FedEx Office Supply Agreement",
        "Type": "Supply",
        "Effective Date": "2021-11-01",
        "Expiration Date": "2023-10-31",
        "Value": "$1,500,000",
        "Status": "Expired"
    },
    {
        "Contract Name": "FedEx Technology Services Contract",
        "Type": "Technology",
        "Effective Date": "2023-07-01",
        "Expiration Date": "2025-06-30",
        "Value": "$5,800,000",
        "Status": "Active"
    }
]

def main():
    st.set_page_config(layout="wide", page_title="FedEx ContractIQ")
    
    # Apply FedEx color scheme
    st.markdown(f"""
        <style>
            /* Main styles */
            body {{
                color: {TEXT_COLOR};
                background-color: {BACKGROUND_COLOR};
            }}
            
            /* Header */
            .header {{
                background-color: {PRIMARY_COLOR};
                color: white;
                padding: 1rem;
                margin-bottom: 2rem;
            }}
            
            /* Tabs */
            .stTabs [role="tablist"] {{
                border-bottom: 2px solid {PRIMARY_COLOR};
            }}
            
            .stTabs [role="tab"][aria-selected="true"] {{
                color: {PRIMARY_COLOR};
                font-weight: bold;
                border-bottom: 3px solid {SECONDARY_COLOR};
            }}
            
            /* Dataframe */
            .dataframe {{
                border: 1px solid #ddd;
            }}
            
            .dataframe th {{
                background-color: {PRIMARY_COLOR} !important;
                color: white !important;
            }}
            
            .dataframe tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            
            /* Sidebar */
            [data-testid="stSidebar"] {{
                background-color: {BACKGROUND_COLOR};
                border-right: 1px solid #ddd;
            }}
            
            /* Buttons */
            .stButton>button {{
                background-color: {SECONDARY_COLOR};
                color: white;
                border: none;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(f"""
        <div class="header">
            <h1 style="margin:0; padding:0;">FedEx ContractIQ</h1>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("Configuration")
        
        # Path Selection
        path_options = ["Local Machine", "Network Path"]
        selected_path = st.selectbox("Source Path", options=path_options)
        
        # AI Model Selection
        ai_model_options = ["Transportation & Logistics", "Warehousing & Storage", "Customer Contracts"]
        selected_model = st.selectbox("AI Model", options=ai_model_options)
        
        # File Uploader
        uploaded_files = st.file_uploader(
            "Upload Contract Files", 
            type=["pdf"], 
            accept_multiple_files=True
        )

    # Main Content
    if uploaded_files:
        num_files = len(uploaded_files)
        if num_files > len(SAMPLE_CONTRACTS):
            st.warning(f"Showing first {len(SAMPLE_CONTRACTS)} sample contracts")
            num_files = len(SAMPLE_CONTRACTS)

        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "Critical Data Insights", 
            "Commercial Insights", 
            "Legal Insights"
        ])

        with tab1:
            # Display only the number of contracts matching uploaded files
            df = pd.DataFrame(SAMPLE_CONTRACTS[:num_files])
            st.dataframe(df, use_container_width=True)
            
            # Donut chart of contract types
            if num_files > 0:
                type_counts = df['Type'].value_counts().reset_index()
                type_counts.columns = ['Type', 'Count']
                
                fig = px.pie(
                    type_counts, 
                    values='Count', 
                    names='Type',
                    hole=0.4,
                    color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR]
                )
                fig.update_layout(
                    title="Contract Type Distribution",
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Commercial data would go here
            st.write("Commercial insights data would be displayed here")

        with tab3:
            # Legal data would go here
            st.write("Legal insights data would be displayed here")

        # Chat Interface
        st.divider()
        st.subheader("Contract Assistant")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about your contracts"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Simulated AI response
            response = f"Sample response regarding {prompt}"
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
