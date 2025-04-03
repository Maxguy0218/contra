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

# Complete dataset definitions
CRITICAL_DATA = {
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
                     "Cybersecurity Services (Vulnerability Mgmt)", "Managed Detection and Response (MDR)", 
                     "Microsoft Azure Cloud Services"],
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

COMMERCIAL_DATA = { 
    # ... (same commercial data structure as before)
}

LEGAL_DATA = {
    # ... (same legal data structure as before)
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
    st.set_page_config(layout="wide", page_title="ContractIQ - FedEx", page_icon="📄")
    
    # Custom CSS with FedEx styling
    st.markdown(f"""
        <style>
            /* Main container styling */
            .main {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
            }}
            
            /* Header styling */
            .header-container {{
                text-align: center;
                margin: -50px 0 -20px 0;
                padding: 20px 0;
                background-color: {PRIMARY_COLOR};
                border-radius: 0 0 15px 15px;
            }}
            
            .main-title {{
                font-size: 2.2rem;
                font-weight: 700;
                color: {BACKGROUND_COLOR};
                display: inline-block;
                vertical-align: middle;
                margin: 0;
                letter-spacing: 0.5px;
            }}
            
            /* Tab styling */
            .stTabs [role=tablist] {{
                display: flex;
                justify-content: center;
                gap: 10px;
                margin: 0 auto 30px;
                padding: 12px;
                background: {BACKGROUND_COLOR};
                border-bottom: 2px solid {PRIMARY_COLOR};
            }}
            
            .stTabs [role=tab] {{
                padding: 12px 24px;
                border-radius: 4px;
                background: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
                font-weight: 500;
                font-size: 0.9rem;
                border: none;
                transition: all 0.3s ease;
                border-bottom: 2px solid transparent;
            }}
            
            .stTabs [role=tab]:hover {{
                color: {PRIMARY_COLOR};
                border-bottom: 2px solid {SECONDARY_COLOR};
            }}
            
            .stTabs [role=tab][aria-selected=true] {{
                color: {PRIMARY_COLOR};
                font-weight: 600;
                border-bottom: 2px solid {PRIMARY_COLOR};
            }}
            
            /* Dataframe styling */
            .dataframe {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR} !important;
                border: 1px solid #ddd;
            }}
            
            .dataframe th {{
                background-color: {PRIMARY_COLOR} !important;
                color: {BACKGROUND_COLOR} !important;
                font-weight: 600;
            }}
            
            .dataframe tr:nth-child(even) {{
                background-color: #f8f8f8;
            }}
            
            .dataframe tr:hover {{
                background-color: #f0f0f0 !important;
            }}
            
            /* Sidebar styling */
            .sidebar .sidebar-content {{
                background-color: {BACKGROUND_COLOR};
                border-right: 1px solid #ddd;
            }}
            
            .sidebar-title {{
                font-size: 1.4rem;
                color: {PRIMARY_COLOR};
                font-weight: 600;
                margin: -10px 0 20px 0;
                text-align: center;
                padding-bottom: 10px;
                border-bottom: 2px solid {SECONDARY_COLOR};
            }}
            
            .dropdown-section {{
                padding: 18px 0;
                margin: 20px 0;
            }}
            
            .stSelectbox div[data-baseweb="select"] {{
                background-color: {BACKGROUND_COLOR} !important;
                border-color: #ddd !important;
                color: {TEXT_COLOR} !important;
            }}
            
            /* File uploader styling */
            .file-uploader {{
                padding: 20px;
                border-radius: 4px;
                margin-top: 25px;
                border: 2px dashed {PRIMARY_COLOR};
                text-align: center;
                background-color: #f8f8f8;
            }}
            
            /* Chat interface styling */
            .chat-header {{
                font-size: 1.4rem;
                color: {PRIMARY_COLOR};
                font-weight: 600;
                margin: 30px 0 15px 0;
                text-align: center;
            }}
            
            .chat-message {{
                margin: 12px 0;
                padding: 16px 20px;
                border-radius: 4px;
                color: {TEXT_COLOR};
                font-size: 0.95rem;
                line-height: 1.5;
                border: 1px solid #ddd;
                background-color: #f8f8f8;
            }}
            
            .user-message {{
                border-left: 4px solid {SECONDARY_COLOR};
                margin-left: 10%;
            }}
            
            .assistant-message {{
                border-left: 4px solid {PRIMARY_COLOR};
                margin-right: 10%;
            }}
            
            /* Input field styling */
            .stTextInput input {{
                background-color: {BACKGROUND_COLOR} !important;
                color: {TEXT_COLOR} !important;
                border: 1px solid #ddd !important;
                border-radius: 4px !important;
                padding: 12px !important;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Header with logo
    logo_base64 = get_base64_image("logo.svg") if os.path.exists("logo.svg") else ""
    st.markdown(f"""
        <div class="header-container">
            <h1 class="main-title">
                <img src="data:image/svg+xml;base64,{logo_base64}" style="height:45px; vertical-align: middle; margin-right:15px;">
                ContractIQ - FedEx
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.markdown(f"""
            <div class="sidebar-title">
                Configuration Panel
            </div>
        """, unsafe_allow_html=True)

        # Path Selection
        path_options = ["Local Machine", "Network Path"]
        selected_path = st.selectbox(
            "Source Path",
            options=path_options,
            index=0
        )

        # AI Model Selection
        ai_model_options = ["Transportation & Logistics", "Warehousing & Storage", "Customer Contracts"]
        selected_model = st.selectbox(
            "AI Model",
            options=ai_model_options,
            index=0
        )

        # File Uploader
        uploaded_files = st.file_uploader(
            "Upload Contract Files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload multiple PDF contracts for analysis"
        )

    # Main Content Area
    if uploaded_files:
        num_files = len(uploaded_files)
        if num_files > 13:  # Cap at sample data size
            num_files = 13
            st.warning("Showing maximum 13 records from sample data")

        # Tabs
        tab1, tab2, tab3 = st.tabs([
            "Critical Data Insights", 
            "Commercial Insights", 
            "Legal Insights"
        ])
        
        with tab1:
            df = pd.DataFrame({k: v[:num_files] for k, v in CRITICAL_DATA.items()})
            st.dataframe(df.style.set_properties(**{
                'background-color': BACKGROUND_COLOR,
                'color': TEXT_COLOR,
                'border': '1px solid #ddd'
            }), use_container_width=True, height=600)
            
            st.markdown("---")
            st.markdown("### Contract Type Distribution")
            donut_chart = create_donut_chart(CRITICAL_DATA, num_files)
            st.plotly_chart(donut_chart, use_container_width=True)

        with tab2:
            df = pd.DataFrame({k: v[:num_files] for k, v in COMMERCIAL_DATA.items()})
            st.dataframe(df.style.set_properties(**{
                'background-color': BACKGROUND_COLOR,
                'color': TEXT_COLOR,
                'border': '1px solid #ddd'
            }), use_container_width=True, height=600)

        with tab3:
            df = pd.DataFrame({k: v[:num_files] for k, v in LEGAL_DATA.items()})
            st.dataframe(df.style.set_properties(**{
                'background-color': BACKGROUND_COLOR,
                'color': TEXT_COLOR,
                'border': '1px solid #ddd'
            }), use_container_width=True, height=600)

        # Chat Interface
        st.markdown("---")
        st.markdown('<div class="chat-header">Contract Assistant</div>', unsafe_allow_html=True)
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Process documents for AI
        if "vector_store" not in st.session_state:
            with st.spinner("Processing documents..."):
                all_text = [process_pdf(f) for f in uploaded_files]
                st.session_state.vector_store = create_vector_store(all_text)

        # Chat input
        question = st.text_input("Ask a question about your contracts:", key="chat_input")
        if question and st.session_state.vector_store:
            with st.spinner("Generating answer..."):
                response = get_answer(question, st.session_state.vector_store)
                st.session_state.chat_history.append(("user", question))
                st.session_state.chat_history.append(("assistant", response))
                st.experimental_rerun()

        # Display chat history
        for role, text in st.session_state.chat_history:
            div_class = "user-message" if role == "user" else "assistant-message"
            st.markdown(f"""
                <div class="chat-message {div_class}">
                    <b>{role.title()}:</b> {text}
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
