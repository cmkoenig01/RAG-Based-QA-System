# RAG-Based QA System

A Retrieval-Augmented Generation (RAG) application that lets users upload a PDF and ask natural-language questions about its content. Answers are grounded exclusively in the uploaded document, with source page citations included in every response.

---

## Architecture

```
PDF Upload
    │
    ▼
PyPDFLoader ──► RecursiveCharacterTextSplitter (1000 chars / 150 overlap)
                        │
                        ▼
              WatsonxEmbeddings (slate-125m-english-rtrvr-v2)
                        │
                        ▼
                 ChromaDB (in-memory vector store)
                        │
              ┌─────────┘
              │  similarity search (top-k = 4)
              ▼
    Retrieved chunks + user query
              │
              ▼
    WatsonxLLM (granite-3-2-8b-instruct)
    RetrievalQA chain (stuff)
              │
              ▼
    Answer + source page citations
              │
              ▼
       Gradio UI
```

**Key design decisions:**
- **Chunking:** 1000-char chunks with 150-char overlap (~15%) preserves cross-boundary context and improves retrieval recall.
- **Embedding model:** `ibm/slate-125m-english-rtrvr-v2` is a retrieval-optimized encoder; `TRUNCATE_INPUT_TOKENS=512` ensures no content is silently dropped.
- **Retrieval:** Top-4 chunks by cosine similarity are stuffed into the LLM prompt.
- **Generation:** `ibm/granite-3-2-8b-instruct` with `MAX_NEW_TOKENS=512` and `TEMPERATURE=0.5` for factual, coherent answers.
- **Source citations:** Page numbers from retrieved chunks are appended to every answer.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Document loading | LangChain `PyPDFLoader` |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | IBM Watsonx (`slate-125m-english-rtrvr-v2`) |
| Vector store | ChromaDB |
| LLM | IBM Watsonx (`granite-3-2-8b-instruct`) |
| Orchestration | LangChain `RetrievalQA` |
| UI | Gradio |

---

## Setup & Run

### Prerequisites

- Python 3.9+
- An [IBM Watsonx](https://www.ibm.com/watsonx) account with API access

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/rag-qa-system.git
cd rag-qa-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env and fill in your Watsonx credentials
```

Your `.env` file should look like:

```env
WATSONX_API_KEY=your_api_key_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_project_id_here
```

> **Note:** This project was originally developed using IBM Watsonx credentials provided through an academic program. You must supply your own credentials to run it locally.

### Run

```bash
python app.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## Example Usage

1. Upload a PDF (research paper, contract, manual, etc.)
2. Type a question in the text box
3. The system retrieves the most relevant passages and generates a grounded answer with page citations

---

## Project Structure

```
rag-qa-system/
├── app.py              # Main application (RAG pipeline + Gradio UI)
├── requirements.txt    # Python dependencies
├── .env.example        # Template for environment variables
├── .gitignore
└── README.md
```

---

## License

[MIT](LICENSE)
