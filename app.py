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

# Data Sources
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
                 color_discrete_sequence=[FEDEX_PURPLE, FEDEX_ORANGE, "#333333"])
    
    fig.update_traces(textposition='inside', 
                     textinfo='percent+label',
                     marker=dict(line=dict(color=BACKGROUND_COLOR, width=2)))
    
    fig.update_layout(
        height=400,
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

def create_geo_bar_chart(df):
    fig = px.bar(df,
                 y='Geographical Scope',
                 x='Total Contract Value',
                 color='Type of Contract',
                 orientation='h',
                 color_discrete_sequence=[FEDEX_PURPLE, FEDEX_ORANGE, "#333333"],
                 title="Contract Value by Geography & Type")
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        title_font=dict(size=18, color=FEDEX_PURPLE),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
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
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
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

def home_page():
    # Main content container with proper spacing
    with st.container():
        # Configuration panel at the very top
        with st.expander("⚙️ CONFIGURATION", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                selected_path = st.selectbox(
                    "Source Path",
                    ["Local Machine", "Network Path"],
                    index=0
                )
            with col2:
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

        # Rest of the content
        if uploaded_files:
            num_records = len(uploaded_files)
            
            def slice_data(data_dict, num_records):
                return {k: v[:num_records] for k, v in data_dict.items() if len(v) >= num_records}
            
            critical_data = slice_data(CRITICAL_DATA, num_records)
            commercial_data = slice_data(COMMERCIAL_DATA, num_records)
            legal_data = slice_data(LEGAL_DATA, num_records)
            
            # Create combined dataframe for visualization
            combined_df = pd.DataFrame({
                **critical_data,
                **{'Total Contract Value': [float(x.replace('$', '').replace(',', '')) 
                   for x in commercial_data['Total Contract Value']]}
            })
            
            tab1, tab2, tab3 = st.tabs([
                "Critical Data Insights", 
                "Commercial Insights", 
                "Legal Insights"
            ])
            
            with tab1:
                if critical_data:
                    # Create two columns for charts
                    col_chart1, col_chart2 = st.columns([1, 1])
                    
                    with col_chart1:
                        if num_records > 0:
                            donut_chart = create_donut_chart(critical_data, num_records)
                            st.plotly_chart(donut_chart, use_container_width=True)
                    
                    with col_chart2:
                        if num_records > 0:
                            bar_chart = create_geo_bar_chart(combined_df)
                            st.plotly_chart(bar_chart, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Show table with exact number of records
                    critical_df = pd.DataFrame(critical_data)
                    st.dataframe(
                        critical_df, 
                        use_container_width=True,
                        height=40 + 35 * num_records,
                        hide_index=True
                    )
                else:
                    st.warning("No critical data available for the selected contracts")

            with tab2:
                if commercial_data:
                    commercial_df = pd.DataFrame(commercial_data)
                    st.dataframe(
                        commercial_df, 
                        use_container_width=True,
                        height=40 + 35 * num_records,
                        hide_index=True
                    )
                else:
                    st.warning("No commercial data available for the selected contracts")

            with tab3:
                if legal_data:
                    legal_df = pd.DataFrame(legal_data)
                    st.dataframe(
                        legal_df, 
                        use_container_width=True,
                        height=40 + 35 * num_records,
                        hide_index=True
                    )
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
                    <div class="{div_class}">
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
            .stApp {margin-top: 0px !important;}
        </style>
    """, unsafe_allow_html=True)

    # Custom CSS with fixed header and working navigation
    st.markdown(f"""
        <style>
            /* Fixed header styling */
            .header-container {{
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 70px;
                background: white;
                z-index: 999;
                display: flex;
                justify-content: center;
                align-items: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            .header-title {{
                font-size: 2.8rem;
                font-weight: 700;
                margin: 0;
                text-align: center;
            }}
            
            .contract-part {{ color: {FEDEX_PURPLE}; }}
            .iq-part {{ color: {FEDEX_ORANGE}; }}
            
            /* Navigation menu */
            .nav-container {{
                position: fixed;
                left: 0;
                top: 70px;
                bottom: 0;
                width: {NAV_WIDTH};
                background-color: {FEDEX_PURPLE};
                z-index: 998;
                padding-top: 20px;
            }}
            
            .nav-button {{
                width: 100%;
                padding: 15px 0;
                background: transparent;
                border: none;
                color: rgba(255,255,255,0.8);
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            
            .nav-button:hover {{
                background-color: {FEDEX_ORANGE};
                color: white !important;
            }}
            
            .nav-button.active {{
                background-color: {FEDEX_ORANGE};
            }}
            
            .nav-icon {{
                font-size: 20px;
                margin-bottom: 5px;
            }}
            
            .main-content {{
                margin-left: {NAV_WIDTH};
                padding-top: 80px;
            }}
            
            /* Configuration panel styling */
            .stExpander {{
                margin-top: 70px !important;
                z-index: 997;
                position: relative;
            }}
            
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
            
            /* Adjust the main content area */
            .stApp > div:first-child {{
                padding-top: 0 !important;
            }}
            
            /* Ensure content starts below header */
            .stApp {{
                padding-top: 70px !important;
            }}
        </style>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">
                <span class="contract-part">Contract</span>
                <span class="iq-part">IQ</span>
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown(f"""
        <div class="nav-container">
            <button class="nav-button {'active' if st.session_state.get('nav', 'home') == 'home' else ''}" 
                onclick="window.parent.postMessage('home', '*')">
                <i class="fas fa-home nav-icon"></i>
                <span>Home</span>
            </button>
            <div style="height: 20px"></div>
            <button class="nav-button {'active' if st.session_state.get('nav') == 'tools' else ''}" 
                onclick="window.parent.postMessage('tools', '*')">
                <i class="fas fa-tools nav-icon"></i>
                <span>Tools</span>
            </button>
            <div style="height: 20px"></div>
            <button class="nav-button {'active' if st.session_state.get('nav') == 'analytics' else ''}" 
                onclick="window.parent.postMessage('analytics', '*')">
                <i class="fas fa-chart-bar nav-icon"></i>
                <span>Analytics</span>
            </button>
        </div>
        <script>
            window.addEventListener('message', function(event) {{
                if (['home','tools','analytics'].includes(event.data)) {{
                    Streamlit.setComponentValue(event.data);
                }}
            }});
        </script>
    """, unsafe_allow_html=True)

    # Handle navigation
    nav_value = st.session_state.get('nav', 'home')
    
    # Main content
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    if nav_value == 'home':
        home_page()
    elif nav_value == 'tools':
        tools_page()
    elif nav_value == 'analytics':
        analytics_page()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
