import streamlit as st
import pandas as pd
import plotly.express as px
import pdfplumber
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Configuration
GEMINI_API_KEY = 'AIzaSyAm_Fx8efZ2ELCwL0ZzZXMDMbrF6StdKsg'
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Color Scheme
FEDEX_PURPLE = "#4D148C"
FEDEX_ORANGE = "#FF6200"
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#333333"
NAV_WIDTH = "60px"

# Data Sources (same as before)
CRITICAL_DATA = { ... }
COMMERCIAL_DATA = { ... }
LEGAL_DATA = { ... }

def create_donut_chart(data, num_records):
    # Same as before

def process_pdf(uploaded_file):
    # Same as before

def create_vector_store(texts):
    # Same as before

def get_answer(question, vector_store):
    # Same as before

def home_page(uploaded_files):
    if uploaded_files:
        num_records = len(uploaded_files)
        
        def slice_data(data_dict, num_records):
            return {k: v[:num_records] for k, v in data_dict.items() if len(v) >= num_records}
        
        critical_data = slice_data(CRITICAL_DATA, num_records)
        commercial_data = slice_data(COMMERCIAL_DATA, num_records)
        legal_data = slice_data(LEGAL_DATA, num_records)
        
        tab1, tab2, tab3 = st.tabs([
            "Critical Data Insights", 
            "Commercial Insights", 
            "Legal Insights"
        ])
        
        with tab1:
            if critical_data:
                critical_df = pd.DataFrame(critical_data)
                st.dataframe(critical_df, use_container_width=True, height=600)
                
                if num_records > 0:
                    st.markdown("---")
                    st.markdown("### Contract Type Distribution")
                    donut_chart = create_donut_chart(critical_data, num_records)
                    st.plotly_chart(donut_chart, use_container_width=True)
            else:
                st.warning("No critical data available for the selected contracts")

        with tab2:
            if commercial_data:
                commercial_df = pd.DataFrame(commercial_data)
                st.dataframe(commercial_df, use_container_width=True, height=600)
            else:
                st.warning("No commercial data available for the selected contracts")

        with tab3:
            if legal_data:
                legal_df = pd.DataFrame(legal_data)
                st.dataframe(legal_df, use_container_width=True, height=600)
            else:
                st.warning("No legal data available for the selected contracts")

        # Chat Interface
        st.markdown("---")
        st.markdown("## Document Assistant")
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "vector_store" not in st.session_state:
            with st.spinner("Processing documents..."):
                all_text = [process_pdf(f) for f in uploaded_files]
                if any(all_text):
                    st.session_state.vector_store = create_vector_store(all_text)
                else:
                    st.error("No text could be extracted from the uploaded documents")
        
        question = st.text_input("Ask a question about your contracts:", key="chat_input")
        if question and st.session_state.get('vector_store'):
            with st.spinner("Generating answer..."):
                response = get_answer(question, st.session_state.vector_store)
                st.session_state.chat_history.append(("user", question))
                st.session_state.chat_history.append(("assistant", response))
                st.experimental_rerun()
        
        for role, text in st.session_state.chat_history:
            div_class = "user-message" if role == "user" else "assistant-message"
            st.markdown(f"""
                <div style="background-color: {FEDEX_PURPLE if role == 'user' else FEDEX_ORANGE}; color: white; padding: 12px; border-radius: 8px; margin: 8px 0;">
                    <b>{role.title()}:</b> {text}
                </div>
            """, unsafe_allow_html=True)

def tools_page():
    st.markdown("## Tools Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Document Analysis Tools")
        st.write("- Contract Comparator")
        st.write("- Clause Library")
        st.write("- Risk Assessor")
    with col2:
        st.subheader("Workflow Automation")
        st.write("- Automated Renewal Tracker")
        st.write("- Compliance Checker")
        st.write("- Obligation Manager")

def analytics_page():
    st.markdown("## Advanced Analytics")
    st.subheader("Contract Portfolio Health")
    data = pd.DataFrame({
        'Metric': ['Risk Score', 'Compliance %', 'Renewal Density', 'Value Concentration'],
        'Value': [65, 88, 42, 78]
    })
    fig = px.bar(data, x='Metric', y='Value', color='Metric',
                 color_discrete_sequence=[FEDEX_PURPLE, FEDEX_ORANGE, "#333333", "#666666"])
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(layout="wide", page_title="ContractIQ", page_icon="📄")
    
    # Remove Streamlit header and footer
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    # Custom CSS with fixed header and configuration panel
    st.markdown(f"""
        <style>
            /* Fixed header */
            .header {{
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 80px;
                background: {BACKGROUND_COLOR};
                z-index: 1000;
                display: flex;
                align-items: center;
                padding: 0 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            .header-title {{
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0 20px;
                flex-grow: 1;
                text-align: center;
            }}
            
            /* Configuration panel */
            .config-container {{
                position: fixed;
                top: 80px;
                left: 0;
                right: 0;
                background: {BACKGROUND_COLOR};
                z-index: 999;
                padding: 10px 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            
            /* Sidebar */
            .sidebar {{
                position: fixed;
                top: 150px;
                left: 0;
                bottom: 0;
                width: {NAV_WIDTH};
                background: {FEDEX_PURPLE};
                z-index: 998;
                padding-top: 20px;
            }}
            
            /* Main content */
            .main-content {{
                margin-top: 150px;
                margin-left: {NAV_WIDTH};
                padding: 20px;
            }}
            
            /* Navigation buttons */
            .nav-button {{
                display: block;
                width: 100%;
                padding: 12px;
                background: transparent;
                border: none;
                color: white;
                text-align: center;
                cursor: pointer;
                transition: 0.2s;
            }}
            
            .nav-button:hover {{
                background: {FEDEX_ORANGE};
            }}
            
            .nav-button.active {{
                background: {FEDEX_ORANGE};
                font-weight: bold;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None

    # Header
    st.markdown(f"""
        <div class="header">
            <h1 class="header-title">
                <span style="color: {FEDEX_PURPLE}">Contract</span>
                <span style="color: {FEDEX_ORANGE}">IQ</span>
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # Configuration panel (always visible)
    with st.container():
        st.markdown("""
            <div class="config-container">
                <div style="display: flex; gap: 20px; align-items: center;">
                    <div style="flex: 1;">
                        <select style="width: 100%; padding: 8px; border-radius: 4px; border: 1px solid #ddd;">
                            <option>Local Machine</option>
                            <option>Network Path</option>
                        </select>
                    </div>
                    <div style="flex: 1;">
                        <select style="width: 100%; padding: 8px; border-radius: 4px; border: 1px solid #ddd;">
                            <option>Transportation & Logistics</option>
                            <option>Warehousing & Storage</option>
                            <option>Customer Contracts</option>
                        </select>
                    </div>
                    <div style="flex: 2;">
                        <input type="file" id="file-upload" style="display: none;" multiple>
                        <button onclick="document.getElementById('file-upload').click()" style="padding: 8px 15px; background: {FEDEX_PURPLE}; color: white; border: none; border-radius: 4px; cursor: pointer;">
                            Upload Contract Files
                        </button>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Navigation sidebar
    st.markdown(f"""
        <div class="sidebar">
            <button class="nav-button {'active' if st.session_state.current_page == 'Home' else ''}" onclick="setPage('Home')">🏠 Home</button>
            <button class="nav-button {'active' if st.session_state.current_page == 'Tools' else ''}" onclick="setPage('Tools')">🛠️ Tools</button>
            <button class="nav-button {'active' if st.session_state.current_page == 'Analytics' else ''}" onclick="setPage('Analytics')">📊 Analytics</button>
        </div>
        
        <script>
            function setPage(page) {{
                Streamlit.setComponentValue(page);
            }}
        </script>
    """, unsafe_allow_html=True)

    # Handle navigation
    if 'nav' in st.session_state:
        st.session_state.current_page = st.session_state.nav
        st.session_state.nav = None
        st.experimental_rerun()

    # Main content
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    if st.session_state.current_page == 'Home':
        home_page(st.session_state.uploaded_files if st.session_state.uploaded_files else [])
    elif st.session_state.current_page == 'Tools':
        tools_page()
    elif st.session_state.current_page == 'Analytics':
        analytics_page()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
