import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import plotly.express as px

# Color Scheme
FEDEX_PURPLE = "#4D148C"
FEDEX_ORANGE = "#FF6200"
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#333333"

# ... [Keep all your data sources and helper functions identical] ...

def main():
    st.set_page_config(layout="wide", page_title="ContractIQ", page_icon="📄")
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"

    # Remove Streamlit default headers/footers and add custom styling
    st.markdown(f"""
        <style>
            #MainMenu {{visibility: hidden;}}
            header {{visibility: hidden;}}
            .stDeployButton {{display:none;}}
            footer {{visibility: hidden;}}
            
            /* Custom header container */
            .header-container {{
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 70px;
                background: {FEDEX_PURPLE};
                z-index: 1001;
                display: flex;
                align-items: flex-start;
                padding: 8px 20px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            /* Title styling */
            .app-title {{
                font-size: 24px;
                font-weight: bold;
                margin: 0;
                padding: 0;
                line-height: 1;
                transform: translateY(2px);
            }}
            .app-title span:first-child {{
                color: white;
            }}
            .app-title span:last-child {{
                color: {FEDEX_ORANGE};
            }}
            
            /* Navigation buttons container */
            .nav-buttons {{
                display: flex;
                gap: 10px;
                margin: 0;
                padding: 0;
                transform: translateY(2px);
            }}
            
            /* Custom button styling */
            .nav-btn {{
                color: white !important;
                background: none !important;
                border: none !important;
                padding: 6px 14px !important;
                margin: 0 !important;
                line-height: 1 !important;
                height: 34px;
                display: flex;
                align-items: center;
                border-radius: 4px !important;
                transition: all 0.3s !important;
            }}
            .nav-btn:hover {{
                background: {FEDEX_ORANGE} !important;
            }}
            
            /* Active button state */
            .nav-btn.active {{
                background: {FEDEX_ORANGE} !important;
                font-weight: bold !important;
            }}
            
            /* Main content spacing */
            .main-content {{
                padding-top: 85px;
            }}
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
                })
                
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
