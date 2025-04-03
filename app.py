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
PRIMARY_COLOR = "#4D148C"
SECONDARY_COLOR = "#FF6200"
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#333333"

# Sample Data
CRITICAL_DATA = {
    "Engagement": ["IT Services"]*13,
    "Type of Contract": ["SOW", "SOW", "EULA", "EUSA", "SOW", "SOW", "SOW", 
                        "EULA", "EUSA", "SOW", "EUSA", "EUSA", "EULA"],
    "Contract Coverage": ["FedEx Enterprise"]*13,
    "Geographical Scope": ["US"]*13,
    "Contract Scope": ["Professional Services AI"]*13,
    "Effective From": ["2024-01-01"]*13,
    "Expiry Date": ["2026-12-31"]*13,
    "Status": ["Active"]*13,
    "Auto-renewal": ["Yes"]*13
}

COMMERCIAL_DATA = {
    "Total Contract Value": ["$1,000,000"]*13,
    "Payment Terms": ["Net 30"]*13,
    "Early Payment Discount %": ["2%"]*13,
    "Late Payment Penalty%": ["1.5%"]*13,
    "Volume Discounts": ["5%"]*13,
    "Price Adjustment": ["CPI+1%"]*13
}

LEGAL_DATA = {
    "Governing Law": ["Tennessee"]*13,
    "Jurisdiction": ["Memphis, TN"]*13,
    "Liability Cap": ["Contract Value"]*13,
    "Termination Rights": ["90 Days Notice"]*13,
    "Renewal Terms": ["Automatic"]*13
}

def get_base64_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def process_pdf(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception as e:
        st.error(f"Failed to process PDF: {str(e)}")
        return ""

def create_vector_store(texts):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_texts(texts=texts, embedding=embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store: {str(e)}")
        return None

def get_answer(question, vector_store):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        docs = vector_store.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        return model.generate_content(f"Context:\n{context}\n\nQuestion: {question}").text
    except Exception as e:
        return f"Error: {str(e)}"

def create_donut_chart(data, num_files):
    contract_types = data["Type of Contract"][:num_files]
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
    st.set_page_config(layout="wide", page_title="ContractIQ - FedEx")
    
    # Custom CSS
    st.markdown(f"""
        <style>
            /* Main styling */
            .main {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
            }}
            
            /* Header */
            .header {{
                background: {PRIMARY_COLOR};
                padding: 2rem;
                color: white;
                border-radius: 0 0 20px 20px;
                margin-bottom: 2rem;
            }}
            
            /* Tabs */
            .stTabs [role=tablist] {{
                justify-content: center;
                gap: 1rem;
                border-bottom: 2px solid {PRIMARY_COLOR};
            }}
            
            .stTabs [role=tab] {{
                color: {TEXT_COLOR};
                padding: 1rem 2rem;
                border: none;
                transition: 0.3s;
            }}
            
            .stTabs [role=tab][aria-selected=true] {{
                color: {PRIMARY_COLOR};
                font-weight: bold;
                border-bottom: 3px solid {SECONDARY_COLOR};
            }}
            
            /* Sidebar */
            [data-testid="stSidebar"] {{
                background: {BACKGROUND_COLOR};
                border-right: 1px solid #ddd;
            }}
            
            .sidebar-section {{
                border-left: 4px solid {SECONDARY_COLOR};
                padding-left: 1rem;
                margin: 1.5rem 0;
            }}
            
            /* Dataframes */
            .dataframe {{
                border: 1px solid #ddd;
                border-radius: 8px;
            }}
            
            th {{
                background: {PRIMARY_COLOR} !important;
                color: white !important;
            }}
            
            /* Chat */
            .user-msg {{
                border-left: 4px solid {SECONDARY_COLOR};
                padding-left: 1rem;
                margin: 1rem 0;
            }}
            
            .assistant-msg {{
                border-left: 4px solid {PRIMARY_COLOR};
                padding-left: 1rem;
                margin: 1rem 0;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(f"""
        <div class="header">
            <h1 style="margin:0;">
                <img src="data:image/svg+xml;base64,{get_base64_image("fedex-logo.svg") if os.path.exists("fedex-logo.svg") else ""}" 
                     style="height:40px; vertical-align:middle; margin-right:15px;">
                ContractIQ
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## Configuration")
        
        # Source Selection
        with st.container():
            st.markdown("### Source Settings")
            source_type = st.selectbox(
                "Source Type",
                ["Local Machine", "Network Path"],
                index=0
            )
        
        # File Upload
        with st.container():
            st.markdown("### Document Upload")
            uploaded_files = st.file_uploader(
                "Upload Contracts",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload PDF contracts for analysis"
            )

    # Main Content
    if uploaded_files:
        num_files = min(len(uploaded_files), 13)  # Cap at sample data size
        
        # Process documents
        with st.spinner("Processing documents..."):
            texts = [process_pdf(f) for f in uploaded_files]
            vector_store = create_vector_store(texts)
        
        # Tabs
        tab1, tab2, tab3 = st.tabs([
            "Critical Data Insights", 
            "Commercial Insights", 
            "Legal Insights"
        ])
        
        with tab1:
            # Critical Data Table
            critical_df = pd.DataFrame({k: v[:num_files] for k, v in CRITICAL_DATA.items()})
            st.dataframe(critical_df, use_container_width=True)
            
            # Donut Chart
            st.markdown("---")
            st.markdown("### Contract Type Distribution")
            st.plotly_chart(create_donut_chart(CRITICAL_DATA, num_files), use_container_width=True)
        
        with tab2:
            commercial_df = pd.DataFrame({k: v[:num_files] for k, v in COMMERCIAL_DATA.items()})
            st.dataframe(commercial_df, use_container_width=True)
        
        with tab3:
            legal_df = pd.DataFrame({k: v[:num_files] for k, v in LEGAL_DATA.items()})
            st.dataframe(legal_df, use_container_width=True)
        
        # Chat Interface
        st.markdown("---")
        st.markdown("## Contract Assistant")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about the contracts"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                response = get_answer(prompt, vector_store)
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        st.info("Please upload contracts using the sidebar to begin analysis")

if __name__ == "__main__":
    main()
