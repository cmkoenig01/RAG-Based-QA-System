# RAG QA System

A Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask natural language questions using IBM Watsonx and LangChain.

---

## Features
- Upload PDF documents  
- Chunk and embed text using embeddings  
- Store vectors using ChromaDB  
- Query documents using an LLM  
- Interactive Gradio interface  

---

## Tech Stack
- Python  
- IBM Watsonx  
- LangChain  
- ChromaDB  
- Gradio  

---

## ⚙️ Setup, Environment, and Run Instructions

Follow the steps below to install dependencies, configure environment variables, and run the application.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/rag-qa-system.git
cd rag-qa-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create environment variables file
cp .env.example .env

# 4. Add your credentials to the .env file
# (DO NOT commit this file)
WATSONX_API_KEY=your_api_key_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_project_id

# NOTE:
# This project was developed using IBM Watsonx credentials provided through an academic program (Coursera).
# API credentials are not included for security reasons.
# To run the application locally, you must provide your own Watsonx credentials.

# 5. Run the application
python app.py
