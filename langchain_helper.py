
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

api_key = os.getenv("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(model = "gemini-3.8-flash", google_api_key = api_key, temperature = 0.8)

instructor_embedding = HuggingFaceEmbeddings()
vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path="codebasics_faqs.csv", source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embedding)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    #load the vector database from local foldere
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embedding, allow_dangerous_deserialization=True)

    #create a retriever for querying the vector database
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":3})

    prompt_template = """Answer the question using only the context provided below.

    The context contains FAQ records with a prompt and a response.
    Use the response fields as the main source of information.
    Preserve the original response wording as much as possible.

    If the context does not contain enough information to answer a
    part of the question, say "I don't know" for that part.
    Do not invent information.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables={"context", "question"}
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": prompt})

    return chain



if __name__=="__main__":
    chain = get_qa_chain()

    print(chain("do you provide internship? do you guys have EMI option"))
