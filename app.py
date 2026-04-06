import os
import warnings

import gradio as gr
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

load_dotenv()

# Suppress noisy third-party deprecation warnings without nuking all warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ibm_watsonx_ai")


def get_llm() -> WatsonxLLM:
    """Initialize and return the IBM Watsonx LLM.

    Reads WATSONX_API_KEY, WATSONX_URL, and WATSONX_PROJECT_ID from the
    environment (or a .env file).
    """
    parameters = {
        GenParams.MAX_NEW_TOKENS: 512,
        GenParams.TEMPERATURE: 0.5,
    }
    return WatsonxLLM(
        model_id="ibm/granite-3-2-8b-instruct",
        url=os.environ["WATSONX_URL"],
        apikey=os.environ["WATSONX_API_KEY"],
        project_id=os.environ["WATSONX_PROJECT_ID"],
        params=parameters,
    )


def load_document(file_path: str) -> list:
    """Load a PDF from disk and return a list of LangChain Document objects."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(documents: list) -> list:
    """Split documents into overlapping chunks for embedding.

    Uses a 1000-character chunk size with 150-character overlap (~15%) so
    that sentences straddling chunk boundaries are captured by both chunks,
    improving retrieval recall.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    return splitter.split_documents(documents)


def get_embedding_model() -> WatsonxEmbeddings:
    """Initialize and return the IBM Watsonx embedding model.

    Uses slate-125m-english-rtrvr-v2, a retrieval-optimized encoder.
    TRUNCATE_INPUT_TOKENS is set to 512 to handle chunks up to that length
    without silently discarding content.
    """
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": False},
    }
    return WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url=os.environ["WATSONX_URL"],
        apikey=os.environ["WATSONX_API_KEY"],
        project_id=os.environ["WATSONX_PROJECT_ID"],
        params=embed_params,
    )


def build_vector_store(chunks: list) -> Chroma:
    """Embed document chunks and store them in an in-memory Chroma vector store."""
    embedding_model = get_embedding_model()
    return Chroma.from_documents(documents=chunks, embedding=embedding_model)


def build_retriever(file_path: str):
    """Load a PDF, chunk it, embed it, and return a Chroma retriever.

    Returns the top-4 most semantically similar chunks for any query (k=4).
    """
    documents = load_document(file_path)
    chunks = split_documents(documents)
    vector_store = build_vector_store(chunks)
    return vector_store.as_retriever(search_kwargs={"k": 4})


def answer_question(file_path: str, query: str) -> str:
    """Run a RAG query against the uploaded PDF and return the answer.

    Args:
        file_path: Path to the uploaded PDF file.
        query: The natural-language question to answer.

    Returns:
        A string answer grounded in the document's content, or an error message.
    """
    if not file_path:
        return "Please upload a PDF file before asking a question."
    if not query or not query.strip():
        return "Please enter a question."

    try:
        llm = get_llm()
        retriever = build_retriever(file_path)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        result = qa_chain.invoke(query)
        answer = result["result"]

        # Append source page references so answers are citable
        source_docs = result.get("source_documents", [])
        if source_docs:
            pages = sorted({
                doc.metadata.get("page", "?") + 1
                for doc in source_docs
                if isinstance(doc.metadata.get("page"), int)
            })
            if pages:
                answer += f"\n\n*Sources: page(s) {', '.join(str(p) for p in pages)}*"

        return answer

    except KeyError as e:
        return f"Configuration error: missing environment variable {e}. Check your .env file."
    except Exception as e:
        return f"An error occurred while processing your request: {e}"


rag_application = gr.Interface(
    fn=answer_question,
    allow_flagging="never",
    inputs=[
        gr.File(
            label="Upload PDF File",
            file_count="single",
            file_types=[".pdf"],
            type="filepath",
        ),
        gr.Textbox(
            label="Your Question",
            lines=2,
            placeholder="Type your question here...",
        ),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="PDF Question Answering with IBM Watsonx",
    description=(
        "Upload a PDF document and ask any question. "
        "The app uses RAG (Retrieval-Augmented Generation) to answer "
        "using only the content of your document."
    ),
)

if __name__ == "__main__":
    rag_application.launch(server_name="127.0.0.1", server_port=7860)
