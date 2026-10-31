import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

st.title("Codebasics QA 🌱")
btn = st.button("Create Knowledgebase")


if btn:
    create_vector_db()
    
question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    invoke = chain(question)

    st.header("Answer: ")
    st.write(invoke["result"])
