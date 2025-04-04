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

# FedEx Color Scheme
FEDEX_PURPLE = "#4D148C"
FEDEX_ORANGE = "#FF6200"
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#333333"

# Data Source
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
    st.set_page_config(layout="wide", page_title="FedEx ContractIQ")
    
    # Initialize session state
    if 'nav' not in st.session_state:
        st.session_state.nav = 'home'
    if 'drawer_open' not in st.session_state:
        st.session_state.drawer_open = True
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    # Custom CSS
    st.markdown(f"""
        <style>
            /* Main content adjustment */
            .main .block-container {{
                padding-top: 2rem;
            }}
            
            /* Header styling */
            .header-container {{
                text-align: center;
                padding: 20px 0;
                background-color: {FEDEX_PURPLE};
                margin-bottom: 30px;
            }}
            
            .main-title {{
                font-size: 2.5rem;
                font-weight: 700;
                color: white;
            }}
            
            .fed-part {{ color: white; }}
            .ex-part {{ color: {FEDEX_ORANGE}; }}
            
            /* Chat message styling */
            .user-message {{
                background-color: {FEDEX_PURPLE};
                color: white;
                padding: 12px;
                border-radius: 8px;
                margin: 8px 0;
            }}
            
            .assistant-message {{
                background-color: {FEDEX_ORANGE};
                color: white;
                padding: 12px;
                border-radius: 8px;
                margin: 8px 0;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Navigation sidebar
    with st.sidebar:
        st.markdown(f"""
            <style>
                .sidebar .sidebar-content {{
                    width: 80px;
                    background-color: {FEDEX_PURPLE};
                }}
                .nav-btn {{
                    width: 100%;
                    margin: 10px 0;
                    padding: 15px 0;
                    color: white !important;
                    border-radius: 5px;
                    transition: all 0.3s;
                }}
                .nav-btn:hover {{
                    background-color: {FEDEX_ORANGE} !important;
                }}
                .nav-btn.active {{
                    background-color: {FEDEX_ORANGE} !important;
                }}
            </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🏠", key="home_btn", help="Home"):
                st.session_state.nav = 'home'
                st.session_state.drawer_open = True
        with col2:
            if st.button("⚙️", key="tools_btn", help="Tools"):
                st.session_state.nav = 'tools'
                st.session_state.drawer_open = False
        with col3:
            if st.button("📊", key="analytics_btn", help="Analytics"):
                st.session_state.nav = 'analytics'
                st.session_state.drawer_open = False

    # Configuration drawer (only on home page)
    if st.session_state.nav == 'home' and st.session_state.drawer_open:
        with st.sidebar.expander("⚙️ CONFIGURATION", expanded=True):
            selected_path = st.selectbox(
                "Source Path",
                ["Local Machine", "Network Path"],
                index=0
            )
            selected_model = st.selectbox(
                "AI Model",
                ["Transportation & Logistics", "Warehousing & Storage", "Customer Contracts"],
                index=0
            )
            uploaded_files = st.file_uploader(
                "Upload Contract Files",
                type=["pdf"],
                accept_multiple_files=True
            )

    # Main content area
    st.markdown("""
        <div class="header-container">
            <h1 class="main-title">
                <span class="fed-part">Fed</span><span class="ex-part">Ex</span> 
                <span class="fed-part">ContractIQ</span>
            </h1>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.nav == 'home':
        if 'uploaded_files' in locals() and uploaded_files:
            num_records = len(uploaded_files)
            
            # Create dynamic sliced data
            def slice_data(data_dict, num_records):
                return {k: v[:num_records] for k, v in data_dict.items() if len(v) >= num_records}
            
            critical_data = slice_data(CRITICAL_DATA, num_records)
            commercial_data = slice_data(COMMERCIAL_DATA, num_records)
            legal_data = slice_data(LEGAL_DATA, num_records)
            
            # Tabs
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
            
            if st.session_state.vector_store is None:
                with st.spinner("Processing documents..."):
                    all_text = [process_pdf(f) for f in uploaded_files]
                    if any(all_text):
                        st.session_state.vector_store = create_vector_store(all_text)
                    else:
                        st.error("No text could be extracted from the uploaded documents")

            question = st.text_input("Ask a question about your contracts:", key="chat_input")
            if question and st.session_state.vector_store:
                with st.spinner("Generating answer..."):
                    response = get_answer(question, st.session_state.vector_store)
                    st.session_state.chat_history.append(("user", question))
                    st.session_state.chat_history.append(("assistant", response))
                    st.experimental_rerun()

            for role, text in st.session_state.chat_history:
                div_class = "user-message" if role == "user" else "assistant-message"
                st.markdown(f"""
                    <div class="{div_class}">
                        <b>{role.title()}:</b> {text}
                    </div>
                """, unsafe_allow_html=True)

    elif st.session_state.nav == 'tools':
        st.title("Tools Section")
        st.write("This is tools")

    elif st.session_state.nav == 'analytics':
        st.title("Analytics Section")
        st.write("Analytics functionality coming soon...")

if __name__ == "__main__":
    main()
