# app.py

import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import warnings

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Updated LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.base import BaseLLM
from langchain.chains import RetrievalQA
from langchain.schema import (
    Generation,
    LLMResult,
)
from typing import List

# Together AI client library
from together import Together

# Pydantic for Private Attributes
from pydantic import PrivateAttr

# Suppress specific FutureWarning from transformers (Optional)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# ------------------------------
# Configuration and Setup
# ------------------------------

# Load environment variables from .env file
load_dotenv()

# Logging Configuration
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')

# Rotating File Handler with UTF-8 Encoding
log_handler = RotatingFileHandler("app.log", maxBytes=5*1024*1024, backupCount=2, encoding='utf-8')
log_handler.setFormatter(log_formatter)
log_handler.setLevel(logging.DEBUG)

# Initialize Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(log_handler)

# Console Handler with UTF-8 Encoding
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)
console_handler.setStream(os.sys.stdout)  # Ensure it writes to stdout
logger.addHandler(console_handler)

logger.info("Starting application...")

# ------------------------------
# Flask App Initialization
# ------------------------------

app = Flask(__name__)

# CORS Configuration
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5000")  # Update as needed
CORS(app, resources={
    r"/ask_pdf": {
        "origins": FRONTEND_ORIGIN,
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    },
    r"/": {
        "origins": FRONTEND_ORIGIN,
        "methods": ["GET"],
    }
})

# ------------------------------
# Paths Configuration
# ------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, 'pdf')
DB_FOLDER = os.path.join(BASE_DIR, 'db')

# Ensure directories exist
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

# ------------------------------
# Together AI API Configuration
# ------------------------------

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")  # Your API key

if not TOGETHER_API_KEY:
    logger.error("TOGETHER_API_KEY environment variable not set.")
    raise ValueError("TOGETHER_API_KEY environment variable not set.")
else:
    logger.info("TOGETHER_API_KEY loaded successfully.")

# ------------------------------
# CustomLLM Class Definition
# ------------------------------

class CustomLLM(BaseLLM):
    """
    A custom Language Learning Model (LLM) that integrates with the Together API.
    """
    api_key: str
    model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    temperature: float = 0.7

    # Define '_client' as a private attribute to prevent Pydantic from validating it
    _client: Together = PrivateAttr()

    def __init__(self, **kwargs):
        """
        Initializes the CustomLLM with the Together API client.

        Args:
            **kwargs: Accepts 'api_key', 'model_name', and 'temperature'.
        """
        super().__init__(**kwargs)
        self._client = Together(api_key=self.api_key)

    def _generate(self, prompts: List[str], **kwargs) -> LLMResult:
        """
        Generates responses for a list of prompts using the Together API.

        Args:
            prompts (List[str]): A list of input prompts.

        Returns:
            LLMResult: A result containing generations corresponding to each prompt.
        """
        responses = []
        for prompt in prompts:
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=self.temperature,
                    top_p=0.7,
                    top_k=50,
                    repetition_penalty=1,
                    stop=["<|eot_id|>", "<|eom_id|>"],
                    safety_model="meta-llama/Meta-Llama-Guard-3-8B"
                )
                responses.append(Generation(text=response.choices[0].message.content))
            except Exception as e:
                # Log the error and append a fallback message
                logger.error(f"Error generating response for prompt '{prompt}': {e}")
                responses.append(Generation(text="Sorry, I couldn't process that request."))
        return LLMResult(generations=[[generation] for generation in responses])

    @property
    def _llm_type(self) -> str:
        """
        Returns the type of the LLM.

        Returns:
            str: The type identifier of the LLM.
        """
        return "custom_together_llm"

# ------------------------------
# Initialize Embeddings
# ------------------------------

try:
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    logger.info("HuggingFaceEmbeddings initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize HuggingFaceEmbeddings: {e}")
    raise

# ------------------------------
# Initialize Vector Store
# ------------------------------

try:
    vector_store = Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=embedding,
    )
    logger.info("Chroma vector store initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Chroma vector store: {e}")
    raise

# ------------------------------
# Initialize CustomLLM Instance
# ------------------------------

try:
    llm = CustomLLM(
        api_key=TOGETHER_API_KEY,
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        temperature=0.7
    )
    logger.info("CustomLLM initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize CustomLLM: {e}")
    raise

# ------------------------------
# Initialize RetrievalQA Chain
# ------------------------------

try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20}  # Adjust 'k' as needed
        ),
        return_source_documents=False  # Disable source documents
    )
    logger.info("RetrievalQA chain initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize RetrievalQA chain: {e}")
    raise

# ------------------------------
# Configure Text Splitter
# ------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False
)

# ------------------------------
# PDF Processing Functions
# ------------------------------

def process_pdf(file_path: str, file_name: str):
    """
    Processes a single PDF file: loads, splits, and adds to the vector store.

    Args:
        file_path (str): Path to the PDF file.
        file_name (str): Name of the PDF file.
    """
    try:
        logger.info(f"Processing PDF: {file_name}")

        # Load and split the PDF
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents from PDF.")

        # Update metadata with source filename
        for doc in docs:
            doc.metadata['source'] = file_name

        # Split documents into chunks
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks.")

        # Log sample chunks for verification
        for i, chunk in enumerate(chunks[:3]):  # Log first 3 chunks
            logger.debug(f"Chunk {i+1}: {chunk.page_content[:100]}...")

        # Add documents to Chroma
        vector_store.add_documents(chunks)
        vector_store.persist()
        logger.info(f"Added documents to Chroma and persisted at {DB_FOLDER}.")

    except Exception as e:
        logger.error(f"Error processing PDF {file_name}: {e}")

def process_existing_pdfs():
    """
    Processes all existing PDFs in the PDF_FOLDER.
    Skips PDFs that have already been processed.
    """
    logger.info("Processing existing PDFs in the pdf folder.")
    try:
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDFs to process.")
    except Exception as e:
        logger.error(f"Failed to list PDFs in {PDF_FOLDER}: {e}")
        return

    if not pdf_files:
        logger.info("No existing PDFs to process.")
        return

    # Retrieve existing document sources to avoid reprocessing
    existing_sources = set()
    try:
        # Access the underlying Chroma collection
        collection = vector_store._collection  # Note: Accessing a protected member
        all_documents = collection.get(include=["metadatas", "documents"])

        for metadata in all_documents['metadatas']:
            if metadata and 'source' in metadata:
                existing_sources.add(metadata['source'])
        logger.info(f"Existing sources in Chroma: {existing_sources}")
    except Exception as e:
        logger.warning(f"Could not retrieve existing documents from vector store: {e}")

    for file_name in pdf_files:
        if file_name in existing_sources:
            logger.info(f"Skipping already processed PDF: {file_name}")
            continue

        file_path = os.path.join(PDF_FOLDER, file_name)
        process_pdf(file_path, file_name)

# ------------------------------
# Response Generation Function
# ------------------------------

def generate_response(query: str) -> str:
    """
    Generates a response to the user's query using the RetrievalQA chain.

    Args:
        query (str): The user's query.

    Returns:
        str: The generated response.
    """
    try:
        logger.info(f"Generating response for query: {query}")
        answer = qa_chain.run(query)
        logger.info(f"LLM response: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        return "Sorry, I couldn't process that request."

# ------------------------------
# Flask Routes
# ------------------------------

@app.route("/", methods=["GET"])
def home():
    """
    Serves the chatbot's main HTML page.
    """
    return render_template("index.html")

@app.route("/ask_pdf", methods=["POST"])
def ask_pdf():
    """
    Handles user queries sent via POST requests.
    Expects JSON data with a 'query' field.

    Returns:
        JSON response containing the chatbot's answer or an error message.
    """
    try:
        data = request.get_json()
        user_query = data.get("query", "").strip()

        if not user_query:
            return jsonify({"error": "Empty query provided."}), 400

        logger.info(f"Received user query: {user_query}")

        # Generate response using the RetrievalQA chain
        answer = generate_response(user_query)

        return jsonify({"response": answer}), 200
    except Exception as e:
        logger.error(f"Error in /ask_pdf route: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

# ------------------------------
# Application Entry Point
# ------------------------------

def main():
    """
    Entry point of the application. Processes existing PDFs and starts the Flask server.
    """
    try:
        # Process existing PDFs on startup
        process_existing_pdfs()
        logger.info("PDF processing completed.")

        # Run the Flask app
        app.run(host="0.0.0.0", port=5000, debug=False)

    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

# ------------------------------
# Entry Point Check
# ------------------------------

if __name__ == "__main__":
    main()
