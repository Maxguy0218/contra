import streamlit as st
import pandas as pd
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

def main():
    st.set_page_config(layout="wide")
    
    # Custom CSS with enhanced styling
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
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .stTabs [role=tablist] {
                display: flex;
                justify-content: center;
                gap: 15px;
                margin-bottom: 30px;
                background: linear-gradient(90deg, #2d3436 0%, #404040 100%);
                padding: 10px;
                border-radius: 15px;
            }
            .stTabs [role=tab] {
                padding: 12px 25px;
                border-radius: 8px;
                background: #4a4a4a;
                color: white;
                font-weight: bold;
                border: none;
                transition: all 0.3s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            .stTabs [role=tab]:hover {
                background: #FF4500;
                transform: translateY(-2px);
            }
            .stTabs [role=tab][aria-selected=true] {
                background: linear-gradient(145deg, #FF4500, #FF6347);
                color: white;
                box-shadow: 0 4px 6px rgba(255,69,0,0.3);
            }
            .dataframe {
                background-color: #2d3436;
                color: white;
                border-radius: 10px;
                overflow: hidden;
            }
            .sidebar-title {
                font-size: 24px;
                color: #FF4500;
                font-weight: bold;
                margin: -10px 0 20px 0;
                text-align: center;
            }
            .dropdown-section {
                background: #2d3436;
                padding: 15px;
                border-radius: 10px;
                margin: 15px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .chat-message {
                margin: 10px 0;
                padding: 15px;
                border-radius: 10px;
                color: white;
            }
            .user-message {
                background: linear-gradient(145deg, #FF4500, #FF6347);
            }
            .assistant-message {
                background: linear-gradient(145deg, #2d3436, #404040);
            }
            .file-uploader {
                background: #2d3436;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Header
    logo_base64 = get_base64_image("logo.svg") if os.path.exists("logo.svg") else ""
    st.markdown(f"""
        <div class="header-container">
            <h1 class="main-title">
                <img src="data:image/svg+xml;base64,{logo_base64}" style="height:50px; vertical-align: middle">
                ContractIQ
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar with enhanced dropdowns
    with st.sidebar:
        st.markdown(f"""
            <div class="sidebar-title">
                <img src="data:image/svg+xml;base64,{logo_base64}" style="height:40px; vertical-align:middle">
                Configuration
            </div>
        """, unsafe_allow_html=True)

        # Dropdown sections
        st.markdown('<div class="dropdown-section">', unsafe_allow_html=True)
        
        # Path Selection
        path_options = ["Local Machine", "Network Path"]
        selected_path = st.selectbox(
            "Path",
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
        
        st.markdown('</div>', unsafe_allow_html=True)

        # File Uploader
        st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload Contract(s)", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Upload multiple PDF contracts for analysis"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Main Content
    if uploaded_files:
        # Centered Tabs with icons
        tab1, tab2, tab3 = st.tabs([
            "📊 Critical Data Insights", 
            "💰 Commercial Insights", 
            "⚖️ Legal Insights"
        ])
        
        num_files = min(len(uploaded_files), 13)
        
        with tab1:
            df = pd.DataFrame({k: v[:num_files] for k, v in CRITICAL_DATA.items()})
            st.dataframe(df.style.set_properties(**{'background-color': '#2d3436', 'color': 'white'}), 
                        use_container_width=True, height=600)

        with tab2:
            df = pd.DataFrame({k: v[:num_files] for k, v in COMMERCIAL_DATA.items()})
            st.dataframe(df.style.set_properties(**{'background-color': '#2d3436', 'color': 'white'}), 
                        use_container_width=True, height=600)

        with tab3:
            df = pd.DataFrame({k: v[:num_files] for k, v in LEGAL_DATA.items()})
            st.dataframe(df.style.set_properties(**{'background-color': '#2d3436', 'color': 'white'}), 
                        use_container_width=True, height=600)

        # Chat Interface
        st.markdown("---")
        st.markdown("### 🤖 Document Assistant")
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Process documents for AI
        if "vector_store" not in st.session_state:
            with st.spinner("Processing documents..."):
                all_text = [process_pdf(f) for f in uploaded_files]
                st.session_state.vector_store = create_vector_store(all_text)

        # Chat input
        question = st.text_input("Ask about your contracts:")
        if question and st.session_state.vector_store:
            with st.spinner("Generating answer..."):
                response = get_answer(question, st.session_state.vector_store)
                st.session_state.chat_history.append(("user", question))
                st.session_state.chat_history.append(("assistant", response))

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
