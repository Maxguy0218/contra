import streamlit as st
import pandas as pd
import base64
import pdfplumber
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Hardcoded API Key (Replace with your actual key)
GEMINI_API_KEY = 'AIzaSyAm_Fx8efZ2ELCwL0ZzZXMDMbrF6StdKsg'

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Predefined data for tables
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
    # ... (rest of critical data columns)
}

COMMERCIAL_DATA = {
    "Total Contract Value": ["$5,250,785", "$6,953,977", "$2,400,000", "$4,750,000",
                           "$6,200,202", "$1,850,000", "$3,309,753", "$1,275,050",
                           "$7,500,060", "$4,409,850", "$2,750,075", "$3,950,040",
                           "$8,250,070"],
    # ... (rest of commercial data columns)
}

LEGAL_DATA = {
    "Right to Indemnify": ["Yes"] * 13,
    "Right to Assign": ["No"] * 13,
    # ... (rest of legal data columns)
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
    
    # Custom CSS
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
            .stTabs [role=tablist] {
                gap: 10px;
                margin-bottom: 20px;
            }
            .stTabs [role=tab][aria-selected=true] {
                background-color: #FF4500;
                color: white;
            }
            .dataframe {
                background-color: #2d3436;
                color: white;
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

    # Sidebar
    with st.sidebar:
        st.markdown(f"""
            <div style="font-size:24px; color:#FF4500; font-weight:bold; margin:-10px 0 20px 0;">
                <img src="data:image/svg+xml;base64,{logo_base64}" style="height:40px; vertical-align:middle">
                Settings
            </div>
        """, unsafe_allow_html=True)

        # File Upload
        uploaded_files = st.file_uploader(
            "Upload Contract(s)", 
            type=["pdf"], 
            accept_multiple_files=True
        )

    # Main Content
    if uploaded_files:
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "Critical Data Insights", 
            "Commercial Insights", 
            "Legal Insights"
        ])

        num_files = min(len(uploaded_files), 13)  # Cap at 13 records
        
        with tab1:
            df = pd.DataFrame({k: v[:num_files] for k, v in CRITICAL_DATA.items()})
            st.dataframe(df, use_container_width=True)

        with tab2:
            df = pd.DataFrame({k: v[:num_files] for k, v in COMMERCIAL_DATA.items()})
            st.dataframe(df, use_container_width=True)

        with tab3:
            df = pd.DataFrame({k: v[:num_files] for k, v in LEGAL_DATA.items()})
            st.dataframe(df, use_container_width=True)

        # Chat Interface
        st.markdown("---")
        st.markdown("### Document Assistant")
        
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
            response = get_answer(question, st.session_state.vector_store)
            st.session_state.chat_history.append(("user", question))
            st.session_state.chat_history.append(("assistant", response))

        # Display chat history
        for role, text in st.session_state.chat_history:
            st.markdown(f"""
                <div style="margin:10px 0; padding:15px; border-radius:10px; 
                    background-color:{"#2d3436" if role == "assistant" else "#FF4500"};
                    color:white">
                    <b>{role.title()}:</b> {text}
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
