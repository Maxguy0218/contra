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
FEDEX_PURPLE = "#4D148C"
FEDEX_ORANGE = "#FF6200"
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#333333"
BORDER_COLOR = "#DDDDDD"
HIGHLIGHT_COLOR = "#F5F5F5"

# Actual Data Source (as provided)
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

LEGAL_DATA = {
    "Right to Indemnify": ["Yes"] * 13,
    "Right to Assign": ["No"] * 13,
    "Right to Terminate": ["Yes"] * 13,
    "Governing Law": ["Tennessee", "Tennessee", "UK", "Tennessee", "Tennessee",
                     "Tennessee", "Singapore", "Delaware", "Tennessee", "UK",
                     "Delaware", "Singapore", "Delaware"],
    "Liability Limit": ["TCV/ Higher or unl fdir IP/Conf", "", "", "", "", "", 
                       "", "", "", "", "", "", ""]
}

def create_donut_chart(data, num_records):
    contract_types = data["Type of Contract"][:num_records]
    type_counts = pd.Series(contract_types).value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    
    fig = px.pie(type_counts, 
                 values='Count', 
                 names='Type',
                 hole=0.4,
                 title="Contract Type Distribution",
                 color_discrete_sequence=[FEDEX_PURPLE, FEDEX_ORANGE])
    
    fig.update_traces(textposition='inside', 
                     textinfo='percent+label',
                     marker=dict(line=dict(color=BACKGROUND_COLOR, width=2)))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        title_font=dict(size=18, color=FEDEX_PURPLE),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    return fig

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

def main():
    st.set_page_config(layout="wide", page_title="FedEx ContractIQ", page_icon="📄")
    
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
                margin: -50px 0 10px 0;
                padding: 20px 0;
                background-color: {FEDEX_PURPLE};
            }}
            
            .main-title {{
                font-size: 2.2rem;
                font-weight: 700;
                color: {BACKGROUND_COLOR};
                display: inline-block;
                vertical-align: middle;
                margin: 0;
                letter-spacing: 0.5px;
                font-family: 'FedEx Sans', Arial, sans-serif;
            }}
            
            /* Tab styling - moved down by 2px */
            .stTabs [role=tablist] {{
                display: flex;
                justify-content: center;
                gap: 10px;
                margin: 10px auto 30px;
                padding: 12px;
                background: {BACKGROUND_COLOR};
                max-width: 800px;
                border-bottom: 2px solid {FEDEX_PURPLE};
            }}
            
            .stTabs [role=tab] {{
                padding: 10px 20px;
                border-radius: 4px 4px 0 0;
                background: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
                font-weight: 600;
                font-size: 0.9rem;
                border: none;
                transition: all 0.3s ease;
                margin: 0 5px;
                font-family: 'FedEx Sans', Arial, sans-serif;
            }}
            
            .stTabs [role=tab]:hover {{
                color: {FEDEX_PURPLE};
                background-color: {HIGHLIGHT_COLOR};
            }}
            
            .stTabs [role=tab][aria-selected=true] {{
                color: {BACKGROUND_COLOR};
                background-color: {FEDEX_PURPLE};
                border-bottom: 3px solid {FEDEX_ORANGE};
            }}
            
            /* Dataframe styling */
            .dataframe {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                font-family: 'FedEx Sans', Arial, sans-serif;
            }}
            
            .dataframe th {{
                background-color: {FEDEX_PURPLE} !important;
                color: {BACKGROUND_COLOR} !important;
                font-weight: 600;
            }}
            
            .dataframe tr:nth-child(even) {{
                background-color: {HIGHLIGHT_COLOR};
            }}
            
            .dataframe tr:hover {{
                background-color: #EAEAEA !important;
            }}
            
            /* Sidebar styling - removed dotted border */
            .sidebar .sidebar-content {{
                background-color: {BACKGROUND_COLOR};
                border-right: none;
            }}
            
            .sidebar-title {{
                font-size: 1.3rem;
                color: {FEDEX_PURPLE};
                font-weight: 700;
                margin: -10px 0 20px 0;
                text-align: center;
                padding-bottom: 10px;
                border-bottom: 2px solid {FEDEX_ORANGE};
                font-family: 'FedEx Sans', Arial, sans-serif;
            }}
            
            .dropdown-section {{
                padding: 12px 0;
                margin: 15px 0;
            }}
            
            .dropdown-section label {{
                color: {FEDEX_PURPLE} !important;
                font-weight: 600;
                font-family: 'FedEx Sans', Arial, sans-serif;
            }}
            
            .stSelectbox div[data-baseweb="select"] {{
                background-color: {BACKGROUND_COLOR} !important;
                border-color: {BORDER_COLOR} !important;
                color: {TEXT_COLOR} !important;
            }}
            
            /* File uploader styling - removed dotted border */
            .file-uploader {{
                padding: 20px;
                border-radius: 4px;
                margin-top: 20px;
                text-align: center;
            }}
            
            /* Chat interface styling */
            .chat-header {{
                font-size: 1.3rem;
                color: {FEDEX_PURPLE};
                font-weight: 700;
                margin: 30px 0 15px 0;
                text-align: center;
                font-family: 'FedEx Sans', Arial, sans-serif;
            }}
            
            .chat-message {{
                margin: 12px 0;
                padding: 16px 20px;
                border-radius: 4px;
                color: {BACKGROUND_COLOR};
                font-size: 0.95rem;
                line-height: 1.5;
                font-family: 'FedEx Sans', Arial, sans-serif;
            }}
            
            .user-message {{
                background-color: {FEDEX_PURPLE};
                margin-left: 20%;
            }}
            
            .assistant-message {{
                background-color: {FEDEX_ORANGE};
                margin-right: 20%;
            }}
            
            /* Input field styling */
            .stTextInput input {{
                background-color: {BACKGROUND_COLOR} !important;
                color: {TEXT_COLOR} !important;
                border: 1px solid {BORDER_COLOR} !important;
                border-radius: 4px !important;
                padding: 10px 12px !important;
                font-family: 'FedEx Sans', Arial, sans-serif;
            }}
            
            /* Button styling */
            .stButton button {{
                background-color: {FEDEX_PURPLE} !important;
                color: {BACKGROUND_COLOR} !important;
                border: none !important;
                border-radius: 4px !important;
                padding: 10px 20px !important;
                font-weight: 600 !important;
                font-family: 'FedEx Sans', Arial, sans-serif;
            }}
            
            .stButton button:hover {{
                background-color: #3A0C6E !important;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(f"""
        <div class="header-container">
            <h1 class="main-title">
                FedEx ContractIQ
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
        st.markdown('<div class="dropdown-section">', unsafe_allow_html=True)
        path_options = ["Local Machine", "Network Path"]
        selected_path = st.selectbox(
            "Source Path",
            options=path_options,
            index=0
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # AI Model Selection
        st.markdown('<div class="dropdown-section">', unsafe_allow_html=True)
        ai_model_options = ["Transportation & Logistics", "Warehousing & Storage", "Customer Contracts"]
        selected_model = st.selectbox(
            "AI Model",
            options=ai_model_options,
            index=0
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # File Uploader - removed dotted border
        st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload Contract Files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload multiple PDF contracts for analysis"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Main Content Area
    if uploaded_files:
        num_records = len(uploaded_files)
        
        # Tabs - moved down by 2px in CSS
        tab1, tab2, tab3 = st.tabs([
            "Critical Data Insights", 
            "Commercial Insights", 
            "Legal Insights"
        ])
        
        with tab1:
            # Only show the number of rows matching uploaded files
            critical_df = pd.DataFrame({k: v[:num_records] for k, v in CRITICAL_DATA.items()})
            st.dataframe(critical_df, use_container_width=True, height=600)
            
            # Donut chart
            st.markdown("---")
            st.markdown("### Contract Type Distribution")
            donut_chart = create_donut_chart(CRITICAL_DATA, num_records)
            st.plotly_chart(donut_chart, use_container_width=True)

        with tab2:
            # Only show the number of rows matching uploaded files
            commercial_df = pd.DataFrame({k: v[:num_records] for k, v in COMMERCIAL_DATA.items()})
            st.dataframe(commercial_df, use_container_width=True, height=600)

        with tab3:
            # Only show the number of rows matching uploaded files
            legal_df = pd.DataFrame({k: v[:num_records] for k, v in LEGAL_DATA.items()})
            st.dataframe(legal_df, use_container_width=True, height=600)

        # Chat Interface
        st.markdown("---")
        st.markdown('<div class="chat-header">Document Assistant</div>', unsafe_allow_html=True)
        
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
