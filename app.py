import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import plotly.express as px

# Color Scheme
FEDEX_PURPLE = "#4D148C"
FEDEX_ORANGE = "#FF6200"
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#333333"

# Data Sources
CRITICAL_DATA = {
    "Engagement": ["IT - Services", "IT - Services", "IT - Services", "IT - Services", 
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
                 color='Engagement',
                 orientation='h',
                 color_discrete_sequence=px.colors.qualitative.Vivid,
                 title="Contract Value by Geography & Engagement")
    
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

def main():
    st.set_page_config(layout="wide", page_title="ContractIQ", page_icon="📄")
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"

    # Remove Streamlit default headers/footers
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            
            /* Custom header container */
            .header-container {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 70px;
                background: #4D148C;
                z-index: 1001;
                display: flex;
                align-items: center;
                padding: 0 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            /* Title styling */
            .app-title {
                font-size: 24px;
                font-weight: bold;
                margin-right: auto;
            }
            .app-title span:first-child {
                color: white;
            }
            .app-title span:last-child {
                color: #FF6200;
            }
            
            /* Navigation buttons container */
            .nav-buttons {
                display: flex;
                gap: 10px;
            }
            
            /* Custom button styling */
            .nav-btn {
                color: white !important;
                background: none !important;
                border: none !important;
                padding: 8px 16px !important;
                border-radius: 4px !important;
                transition: all 0.3s !important;
            }
            .nav-btn:hover {
                background: #FF6200 !important;
            }
            .nav-btn:active {
                background: #FF6200 !important;
            }
            
            /* Active button state */
            .nav-btn.active {
                background: #FF6200 !important;
                font-weight: bold !important;
            }
            
            /* Main content spacing */
            .main-content {
                padding-top: 90px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Create header container
    header = st.container()
    with header:
        # Create columns for title and navigation
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown(
                '<div class="app-title"><span>Contract</span><span>IQ</span></div>',
                unsafe_allow_html=True
            )
        
        with col2:
            # Create navigation buttons
            cols = st.columns(4)
            with cols[0]:
                home_btn = st.button(
                    "Home",
                    key="home_btn",
                    help="Go to Home page",
                    type="primary" if st.session_state.current_page == "Home" else "secondary"
                )
            with cols[1]:
                history_btn = st.button(
                    "History",
                    key="history_btn",
                    help="Go to History page",
                    type="primary" if st.session_state.current_page == "History" else "secondary"
                )
            with cols[2]:
                playbook_btn = st.button(
                    "Playbook",
                    key="playbook_btn",
                    help="Go to Playbook page",
                    type="primary" if st.session_state.current_page == "Playbook" else "secondary"
                )
            with cols[3]:
                settings_btn = st.button(
                    "Settings",
                    key="settings_btn",
                    help="Go to Settings page",
                    type="primary" if st.session_state.current_page == "Settings" else "secondary"
                )
        
        # Add the fixed header container
        st.markdown('<div class="header-container"></div>', unsafe_allow_html=True)

    # Handle navigation
    if home_btn:
        st.session_state.current_page = "Home"
    if history_btn:
        st.session_state.current_page = "History"
    if playbook_btn:
        st.session_state.current_page = "Playbook"
    if settings_btn:
        st.session_state.current_page = "Settings"

    # Main content
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    if st.session_state.current_page == "Home":
        with st.container():
            # Configuration panel
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
                )
                
                # Permanent Graphs Section
                st.markdown("## Contract Analytics Overview")
                col1, col2, col3 = st.columns([1,1,1])
                
                with col1:
                    donut_chart = create_donut_chart(critical_data, num_records)
                    st.plotly_chart(donut_chart, use_container_width=True)
                    
                with col2:
                    total_values = combined_df.groupby('Engagement')['Total Contract Value'].sum().reset_index()
                    fig_spends = px.bar(combined_df,
                                      x='Engagement',
                                      y='Total Contract Value',
                                      color='Contract Coverage',
                                      title="IT Spends by Engagement",
                                      labels={'Engagement': 'Engagement Type', 'Total Contract Value': 'IT Spends ($)'},
                                      color_discrete_sequence=px.colors.qualitative.Vivid)
                    fig_spends.add_trace(go.Scatter( 
                        x=total_values['Engagement'],
                        y=total_values['Total Contract Value'],
                        text=total_values['Total Contract Value'].apply(lambda x: f"${x:,.0f}"),
                        mode='text',
                        textposition='top center',
                        showlegend=False,
                        textfont=dict(size=12, color=TEXT_COLOR)
                    ))
                    fig_spends.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=50, b=20),
                        title_font=dict(size=18, color=FEDEX_PURPLE),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_spends, use_container_width=True)
                    
                with col3:
                    bar_chart = create_geo_bar_chart(combined_df)
                    st.plotly_chart(bar_chart, use_container_width=True)

                # Tabs Section
                st.markdown("---")
                tab1, tab2, tab3 = st.tabs([
                    "📊 Critical Data Insights", 
                    "💸 Commercial Insights", 
                    "⚖️ Legal Insights"
                ])
                
                with tab1:
                    critical_df = pd.DataFrame(critical_data)
                    st.dataframe(critical_df, use_container_width=True, height=400)
                    
                with tab2:
                    commercial_df = pd.DataFrame(commercial_data)
                    st.dataframe(commercial_df, use_container_width=True, height=400)
                    
                with tab3:
                    legal_df = pd.DataFrame(legal_data)
                    st.dataframe(legal_df, use_container_width=True, height=400)

                # Document Assistant Placeholder
                st.markdown("---")
                st.markdown("## Document Assistant")
                question = st.text_input("Ask a question about your contracts:", key="chat_input")
                
    elif st.session_state.current_page == "History":
        st.markdown("# History")
        st.write("This is the History page placeholder content")
        st.image("https://via.placeholder.com/800x400.png?text=History+Page", use_column_width=True)
        
    elif st.session_state.current_page == "Playbook":
        st.markdown("# Playbook")
        st.write("This is the Playbook page placeholder content")
        st.image("https://via.placeholder.com/800x400.png?text=Playbook+Page", use_column_width=True)
        
    elif st.session_state.current_page == "Settings":
        st.markdown("# Settings")
        st.write("This is the Settings page placeholder content")
        st.image("https://via.placeholder.com/800x400.png?text=Settings+Page", use_column_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
