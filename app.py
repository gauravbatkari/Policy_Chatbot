import streamlit as st
from pdf_reader import read_pdf
from embedding import create_index
from qa_model import get_answer
import os

st.title("📄 Company Policy Q&A Chatbot")

if "indexed" not in st.session_state:
    st.session_state.indexed = False

pdf_path = "Company_Policy.pdf"

if not st.session_state.indexed:
    if os.path.exists(pdf_path):
        text = read_pdf(pdf_path)
        create_index(text)
        st.session_state.indexed = True
        st.success("Company Policy PDF indexed successfully!")

question = st.text_input("Ask a question about company policy:")

if question:
    answer = get_answer(question)
    st.write("**Answer:**", answer)