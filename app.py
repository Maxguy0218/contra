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

# FedEx Color Scheme
PRIMARY_COLOR = "#4D148C"  # Purple
SECONDARY_COLOR = "#FF6200"  # Orange
BACKGROUND_COLOR = "#FFFFFF"  # White
TEXT_COLOR = "#333333"  # Dark Gray

def get_base64_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def process_pdf(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            return {
                "filename": uploaded_file.name,
                "content": "\n".join([page.extract_text() or "" for page in pdf.pages]),
                "pages": len(pdf.pages)
            }
    except Exception as e:
        st.error(f"Failed to process PDF: {str(e)}")
        return None

def extract_contract_data(text):
    # Simplified example - replace with actual ML model integration
    return {
        "Contract Type": "SOW" if "statement of work" in text.lower() else "EULA",
        "Parties": "FedEx and Vendor",
        "Effective Date": "2024-01-01",
        "Term": "3 years",
        "Value": "$1,000,000"
    }

def create_donut_chart(contract_types):
    type_counts = pd.Series(contract_types).value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    
    fig = px.pie(type_counts, 
                 values='Count', 
                 names='Type',
                 hole=0.4,
                 title="Contract Type Distribution",
                 color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR])
    
    fig.update_traces(textposition='inside', 
                     textinfo='percent+label',
                     marker=dict(line=dict(color=BACKGROUND_COLOR, width=2)))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        title_font=dict(size=18, color=PRIMARY_COLOR),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    return fig

def main():
    st.set_page_config(layout="wide", page_title="ContractIQ - FedEx", page_icon="📄")
    
    # Custom CSS Injection
    st.markdown(f"""
        <style>
            /* Main styles */
            .main {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
                font-family: 'Arial', sans-serif;
            }}
            
            /* Header styling */
            .header {{
                background-color: {PRIMARY_COLOR};
                padding: 2rem;
                color: {BACKGROUND_COLOR};
                border-radius: 0 0 20px 20px;
                margin-bottom: 2rem;
            }}
            
            /* Tab styling */
            .stTabs [role=tablist] {{
                border-bottom: 2px solid {PRIMARY_COLOR};
            }}
            
            .stTabs [role=tab] {{
                color: {TEXT_COLOR};
                font-weight: 500;
                padding: 0.8rem 2rem;
                transition: all 0.2s;
            }}
            
            .stTabs [role=tab][aria-selected=true] {{
                color: {PRIMARY_COLOR} !important;
                font-weight: 600;
                border-bottom: 3px solid {SECONDARY_COLOR};
            }}
            
            /* Data table styling */
            .stDataFrame {{
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
            }}
            
            /* Sidebar cleanup */
            [data-testid="stSidebar"] {{
                background: {BACKGROUND_COLOR};
                border-right: 1px solid #ddd;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(f"""
        <div class="header">
            <h1 style="margin:0; font-size:2.2rem;">
                <img src="data:image/svg+xml;base64,{get_base64_image('fedex-logo.svg') if os.path.exists('fedex-logo.svg') else ''}" 
                     style="height:40px; vertical-align:middle; margin-right:15px;">
                ContractIQ
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.selectbox("Data Source", ["Local Machine", "Network Path"], key="source")
        st.selectbox("Analysis Model", ["Transportation", "Warehousing", "Customer"], key="model")
        
        uploaded_files = st.file_uploader(
            "Upload Contracts",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF contracts for analysis"
        )

    # Main Content
    if uploaded_files:
        # Process files
        with st.spinner("Analyzing contracts..."):
            contracts = [process_pdf(f) for f in uploaded_files]
            valid_contracts = [c for c in contracts if c is not None]
            
            # Extract contract data
            contract_data = [extract_contract_data(c["content"]) for c in valid_contracts]
            df = pd.DataFrame(contract_data)
            
            # Create chart data
            contract_types = df["Contract Type"].tolist()

        # Tabs
        tab1, tab2 = st.tabs(["Contract Overview", "Advanced Analytics"])
        
        with tab1:
            # Display processed data
            st.dataframe(
                df,
                use_container_width=True,
                height=600,
                column_config={
                    "Value": st.column_config.NumberColumn(format="$%d")
                }
            )
            
            # Donut chart
            st.markdown("---")
            st.plotly_chart(create_donut_chart(contract_types), use_container_width=True)

        with tab2:
            # Add advanced analytics here
            st.write("Advanced analytics view")

    else:
        st.info("Please upload contracts through the sidebar to begin analysis")

if __name__ == "__main__":
    main()
