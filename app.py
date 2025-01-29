import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.express as px
import base64
import pdfplumber
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Define the paths for the JSON files relative to the script
ATENA_JSON_PATH = os.path.join(os.getcwd(), "atena_annotations_fixed.json")
BCBS_JSON_PATH = os.path.join(os.getcwd(), "bcbs_annotations_fixed.json")

# Load the annotated data from JSON files
def load_atena_data():
    with open(ATENA_JSON_PATH, "r") as f:
        return pd.DataFrame(json.load(f))

def load_bcbs_data():
    with open(BCBS_JSON_PATH, "r") as f:
        return pd.DataFrame(json.load(f))

# Generate key takeaways from the description
def generate_key_takeaways(description):
    sentences = description.split('. ')
    takeaways = [f"• {sentence.strip()}" for sentence in sentences[:5]]
    while len(takeaways) < 5:
        takeaways.append("• [No further details available]")
    return '<br>'.join(takeaways)

# Filter data based on the selected business area
def filter_data(df, business_area):
    df_filtered = df[df["Business Area"] == business_area]
    df_filtered["Key Takeaways"] = df_filtered["Description"].apply(generate_key_takeaways)
    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered[["Term Type", "Sub-Type", "Key Takeaways", "Page #"]]

# Plot pie chart using Plotly
def plot_pie_chart(data, show_labels=True):
    counts = data["Business Area"].value_counts()
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title="Business Area Distribution",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    textinfo = "percent+label" if show_labels else "none"
    fig.update_traces(textinfo=textinfo, pull=[0.1, 0], hole=0.2)
    fig.update_layout(height=600, width=850, margin=dict(l=150, r=150, t=50, b=50))
    return fig

# Function to encode the logo in Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Initialize components for chatbot
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def process_pdf(uploaded_file):
    """Extract and split text from PDF"""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Failed to process PDF: {str(e)}")
        return []

def create_vector_store(texts):
    """Create FAISS vector store with free embeddings"""
    if not texts:
        st.error("No text extracted from document")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_texts(texts=texts, embedding=embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store: {str(e)}")
        return None

def get_answer(question, vector_store, api_key):
    """Get answer using Gemini Pro"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Retrieve relevant text chunks
        docs = vector_store.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Generate answer
        response = model.generate_content(f"Answer based on this context only:\n{context}\n\nQuestion: {question}")
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Streamlit app
def main():
    # Set wide layout
    st.set_page_config(layout="wide")
    
    # Custom Styling with reduced header spacing
    st.markdown("""
        <style>
            .header-container {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: -30px;
            }
            .main-title {
                font-size: 48px;
                font-weight: bold;
                color: #FF5733;
            }
            .sidebar-title {
                font-size: 24px;
                font-weight: bold;
                color: #FF5733;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .report-container {
                max-width: 100%;
                margin: auto;
            }
            .chat-box {
                border: 2px solid #4a4a4a;
                border-radius: 10px;
                padding: 20px;
                height: 400px;
                overflow-y: auto;
                background: #f9f9f9;
                margin-top: 20px;
            }
            .user-msg {
                color: #ffffff;
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                background: #0078d4;
                text-align: right;
            }
            .assistant-msg {
                color: #2d3436;
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                background: #dfe6e9;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Branding in sidebar and main page
    logo_path = "logo.svg"
    if os.path.exists(logo_path):
        logo_base64 = get_base64_image(logo_path)
        logo_img = f'<img src="data:image/svg+xml;base64,{logo_base64}" alt="Logo" style="width: 50px; vertical-align: middle;">'
    else:
        logo_img = "[Logo Missing]"

    st.sidebar.markdown(
        f"""
        <div class="sidebar-title">
            {logo_img}
            ContractIQ
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
        <div class="header-container">
            <img src="data:image/svg+xml;base64,{logo_base64}" alt="Logo" style="width: 120px; vertical-align: middle;">
            <span class="main-title">ContractIQ</span>
        </div>
        """, unsafe_allow_html=True
    )
    
    # Sidebar upload section
    uploaded_file = st.sidebar.file_uploader("Upload a contract file", type=["pdf"])
    gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    
    # Initialize session state
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
        st.session_state.data = None
        st.session_state.business_area = None
        st.session_state.vector_store = None
        st.session_state.messages = []

    # Process document
    if uploaded_file:
        if st.session_state.uploaded_file != uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.data = None
            st.session_state.business_area = None
            
            if "AETNA" in uploaded_file.name.upper():
                st.session_state.data = load_atena_data()
            elif "BLUE" in uploaded_file.name.upper():
                st.session_state.data = load_bcbs_data()
            else:
                st.error("ERROR: Unsupported document type.")
                return
            
            # Process PDF for chatbot
            with st.spinner("Processing document..."):
                texts = process_pdf(uploaded_file)
                st.session_state.vector_store = create_vector_store(texts)
                st.session_state.messages = []  # Reset chat history for new document

    # Report section
    col1, col2 = st.columns([2, 4])
    
    with col1:
        st.subheader("Select Business Area")
        business_area = st.radio(
            "Select a Business Area",
            ["Operational Risk Management", "Financial Risk Management"]
        )
        
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                time.sleep(2)
                report = filter_data(st.session_state.data, business_area)
                st.session_state.report = report
    
    with col2:
        if uploaded_file and st.session_state.data is not None:
            show_labels = not st.sidebar.expander("Options").checkbox("Hide Labels")
            st.write("### Business Area Distribution")
            st.plotly_chart(plot_pie_chart(st.session_state.data, show_labels=show_labels), use_container_width=True)
    
    # Display report
    if "report" in st.session_state and not st.session_state.report.empty:
        st.markdown("<div class='report-container'>", unsafe_allow_html=True)
        st.write(f"### Report for {business_area}")
        st.write(st.session_state.report.to_html(escape=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Chatbot section below report
    st.markdown("---")
    st.subheader("Document Chat Assistant")
    
    # Chat interface
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    
    # User input with auto-clear functionality
    user_input = st.text_input("Ask about the contract:", key="chat_input")
    
    if user_input and st.session_state.vector_store and gemini_api_key:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        try:
            # Get answer from Gemini Pro
            answer = get_answer(user_input, st.session_state.vector_store, gemini_api_key)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Failed to generate answer: {str(e)}")
        
        # Clear input field using callback
        st.session_state.chat_input = ""
        st.experimental_rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
