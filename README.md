# Codebasics Q&A: Question and Answer System Based on Google Palm LLM and Langchain for E-learning company  

This is an end to end LLM project based on Google Palm and Langchain. I build a Q&A system for an e-learning company called codebasics (website: codebasics.io). Codebasics sells data related courses and bootcamps. They have thousands of learners who uses discord server or email to ask questions. This system will provide a streamlit based user interface for students where they can ask questions and get answers. 

<img width="1366" height="630" alt="codebasic Q A" src="https://github.com/user-attachments/assets/c784217b-75fc-4fd8-bc98-e946a36b1dcb" />

## What it does

You type a question, the app finds the FAQ entries that are closest in meaning, and passes them to Gemini along with your question. Gemini is told to answer only from those entries and to say "I don't know" if the answer isn't there. So it won't make up an EMI policy or a course that doesn't exist.

Roughly, the flow looks like this:

```text
codebasics_faqs.csv
   -> CSVLoader
   -> HuggingFace embeddings
   -> FAISS index (saved in faiss_index/)

question -> retriever (top 3 matches) -> prompt -> Gemini -> answer
```
## Features

- Loads FAQ data from a CSV file (`prompt` / `response` columns)
- Semantic search using HuggingFace sentence embeddings stored in a local FAISS index
- Answer generation with Google Gemini via `langchain-google-genai`
- Grounded prompt: answers only from retrieved context, no hallucination
- Simple Streamlit web UI
- Jupyter notebook that walks through the pipeline step by step

## How It Works

```text
codebasics_faqs.csv
        │  CSVLoader
        ▼
   Documents ──► HuggingFace Embeddings ──► FAISS vector store (faiss_index/)
                                                   │
User question ──► Retriever (top-k = 3 similarity) ┘
                          │
                          ▼
              Prompt (context + question)
                          │
                          ▼
                  Gemini LLM (RetrievalQA, "stuff" chain)
                          │
                          ▼
                       Answer
```

1. **Ingest** – `CSVLoader` reads each FAQ row (using `prompt` as the source column).
2. **Embed & index** – rows are embedded with `HuggingFaceEmbeddings` and saved locally with FAISS.
3. **Retrieve** – the top 3 most similar FAQ entries are fetched for the user's question.
4. **Generate** – a `RetrievalQA` chain (`stuff` type) passes the context and question to Gemini using a strict prompt.

## Project Structure

```text
.
├── main.py               # Streamlit app (UI)
├── langchain_helper.py   # Vector DB creation + QA chain logic
├── Q&A_System.ipynb      # Notebook walkthrough / experimentation
├── requirements.txt      # Python dependencies
├── codebasics_faqs.csv   # FAQ knowledge base (not included here — add your own)
├── faiss_index/          # Generated vector store (created at runtime)
└── .env                  # Your API key (not committed)
```

## Setup

You'll need Python 3.10 or newer (I used 3.12) and a Gemini API key from [Google AI Studio](https://aistudio.google.com/).

1.Clone this repository to your local machine using:

```bash
  git clone https://github.com/codebasics/langchain.git
```
2.Navigate to the project directory:

```bash
  cd https://github.com/diksha2023/Q-A-System.git
```
3. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
4.Acquire an api key through makersuite.google.com and put it in .env file

```bash
  GOOGLE_API_KEY="your_api_key_here"

Don't commit this file. Add `.env` and `faiss_index/` to your `.gitignore`.
```
## Usage

1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser.

- To create a knowledebase of FAQs, click on Create Knolwedge Base button. It will take some time before knowledgebase is created so please wait.

- Once knowledge base is created you will see a directory called faiss_index in your current folder

- Now you are ready to ask questions. Type your question in Question box and hit Enter

It opens at [http://localhost:8501](http://localhost:8501). You can also test the chain without the UI by running `python langchain_helper.py`, which fires off a sample question.


## Files

- `main.py`: the Streamlit app
- `langchain_helper.py`: builds the vector index and sets up the QA chain
- `Q&A_System.ipynb`: the notebook where I worked through the pipeline step by step
- `requirements.txt`: dependencies (needs cleanup, see below)
- `codebasics_faqs.csv`: the FAQ data. It isn't included in this repo, so add your own with `prompt` and `response` columns.


## Sample Questions
  - Do you guys provide internship and also do you offer EMI payments?
  - Do you have javascript course?
  - Should I learn power bi or tableau?
  - I've a MAC computer. Can I use powerbi on it?
  - I don't see power pivot. how can I enable it?


