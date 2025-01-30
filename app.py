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

# Hardcoded API Key (Replace with your actual key)
GEMINI_API_KEY = 'AIzaSyAm_Fx8efZ2ELCwL0ZzZXMDMbrF6StdKsg'

# Path configurations
ATENA_JSON_PATH = os.path.join(os.getcwd(), "atena_annotations_fixed.json")
BCBS_JSON_PATH = os.path.join(os.getcwd(), "bcbs_annotations_fixed.json")

def load_atena_data():
    with open(ATENA_JSON_PATH, "r") as f:
        return pd.DataFrame(json.load(f))

def load_bcbs_data():
    with open(BCBS_JSON_PATH, "r") as f:
        return pd.DataFrame(json.load(f))

def filter_data(df, business_area):
    df_filtered = df[df["Business Area"] == business_area]
    return df_filtered[["Term Type", "Sub-Type", "Description", "Page #"]]

def plot_pie_chart(data):
    counts = data["Business Area"].value_counts()
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title="Business Area Distribution",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig.update_traces(textinfo="percent+label", pull=[0.1, 0], hole=0.2)
    fig.update_layout(height=500, width=500, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def process_pdf(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text.split("\n")
    except Exception as e:
        st.error(f"Failed to process PDF: {str(e)}")
        return []

def create_vector_store(texts):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_texts(texts=texts, embedding=embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store: {str(e)}")
        return None

def get_answer(question, vector_store, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        docs = vector_store.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        response = model.generate_content(f"Answer based on this context only:\n{context}\n\nQuestion: {question}")
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def main():
    st.set_page_config(layout="wide")

    # Sidebar
    st.sidebar.header("Upload Contract")
    uploaded_file = st.sidebar.file_uploader("Upload a contract file", type=["pdf"])

    # Branding
    st.markdown('<h1 style="text-align:center;">ContractIQ</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Select Business Area")
        business_area = st.radio("", ["Operational Risk Management", "Financial Risk Management"], label_visibility="collapsed")
        if st.button("Generate Report") and uploaded_file:
            data = load_atena_data() if "AETNA" in uploaded_file.name.upper() else load_bcbs_data()
            st.session_state.report = filter_data(data, business_area)
    
    with col2:
        if uploaded_file:
            data = load_atena_data() if "AETNA" in uploaded_file.name.upper() else load_bcbs_data()
            st.plotly_chart(plot_pie_chart(data), use_container_width=True)

    if "report" in st.session_state:
        st.write(st.session_state.report)
    
    # Chatbot
    st.subheader("Document Chat Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        role_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
        st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)
    
    with st.form(key="chat_form"):
        user_input = st.text_input("Ask about the contract:")
        submit_button = st.form_submit_button("Send")
    
    if submit_button and user_input and uploaded_file:
        texts = process_pdf(uploaded_file)
        vector_store = create_vector_store(texts)
        if vector_store:
            answer = get_answer(user_input, vector_store, GEMINI_API_KEY)
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

if __name__ == "__main__":
    main()
