# RAG (Retrieval-Augmented Generation) Chat Application

This project implements a Retrieval-Augmented Generation (RAG) system using Flask, LangChain, and Ollama. It allows users to upload PDF documents via Postman, ask questions about their content through a chat interface, and receive AI-generated responses based on the document's information.

## Features

- PDF document upload via Postman
- AI-powered question answering based on uploaded documents
- User-friendly chat interface with dark/light theme toggle
- Quick reply suggestions
- Responsive design for various screen sizes

## Directory Structure

```
E:.
│ .env
│ app.py
│ README.md
│ requirements.txt
│
├───db
├───pdf
│ alice.pdf
│
├───static
│ ├───css
│ │ styles.css
│ │
│ ├───images
│ │ Acharya.ai-Finance for growth.png
│ │ corporate-user-icon.png
│ │ moon-icon.png
│ │ sun-icon.png
│ │
│ └───js
│ script.js
│
└───templates
    index.html
```

## Setup and Installation

1. Clone this repository to your local machine.

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables in the `.env` file.

5. Ensure you have Ollama installed and running on your system.

## Running the Application

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:8080` to access the chat interface.

## Usage

1. Upload PDF documents using Postman (see "Uploading PDFs via Postman" section below).
2. Open a web browser and navigate to `http://localhost:8080` to access the chat interface.
3. Ask questions about the uploaded documents in the chat.
4. The AI will provide answers based on the content of the uploaded PDFs.
5. Use quick replies for common follow-up questions.
6. Toggle between dark and light themes using the theme icon.

## How It Works

### PDF Upload and Processing

1. PDFs are uploaded using Postman to the `/pdf` endpoint.
2. When a PDF is uploaded, the application uses the `PDFPlumberLoader` from LangChain to load the PDF content.
3. The loaded content is then split into smaller chunks using the `RecursiveCharacterTextSplitter`. This splitting process creates manageable pieces of text that can be efficiently processed and searched.
4. These chunks are then embedded using the HuggingFace embeddings model (`all-MiniLM-L6-v2`) and stored in a Chroma vector store.

Here's the relevant code snippet from `app.py`:

```python
@app.route("/pdf", methods=["POST"])
def pdf_post():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    file_name = file.filename
    save_file = os.path.join(pdf_folder, file_name)
    file.save(save_file)
    print(f"Saved PDF: {file_name}")
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"Loaded {len(docs)} documents from PDF")
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )
    vector_store.persist()
    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return jsonify(response)
```

### Querying and Retrieval

1. When a user submits a query, the application uses the Chroma vector store to find the most relevant chunks of text based on the query's similarity to the stored embeddings.
2. The retriever is set up with a similarity score threshold to ensure only relevant information is retrieved.
3. The retrieved chunks are then passed to a language model (Llama 2 via Ollama) along with the user's query to generate a contextually relevant response.

Here's the relevant code snippet from `app.py`:

```python
@app.route("/ask_pdf", methods=["POST"])
def ask_pdf_post():
    query = request.json.get("query")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
    
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )
    
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": query})
```

### Uploading PDFs via Postman

To upload a PDF:

1. Open Postman.
2. Create a new POST request to `http://localhost:8080/pdf`.
3. In the request body, select "form-data".
4. Add a key named "file" and set its type to "File".
5. Select the PDF file you want to upload as the value for this key.
6. Send the request.

The server will process the PDF and return a JSON response with details about the upload, including the number of documents and chunks created.

Note: The application saves the PDF files in the `pdf` folder and processes them immediately. Ensure you have the necessary permissions and storage capacity for handling the uploaded files.

## Key Components

- `app.py`: Main Flask application file containing server-side logic.
- `index.html`: Main HTML template for the chat interface.
- `script.js`: Client-side JavaScript for handling user interactions and API calls.
- `styles.css`: CSS file for styling the chat interface.
- `requirements.txt`: List of Python dependencies.

## Technologies Used

- Backend: Flask, LangChain, Ollama, Chroma
- Frontend: HTML, CSS, JavaScript
- AI Model: Llama 2 (via Ollama)
- Embeddings: HuggingFace's all-MiniLM-L6-v2


'''
(venv) PS E:\RAG> python app.py
2024-09-30 15:01:33,866 INFO __main__: Starting application...
2024-09-30 15:01:33,868 INFO __main__: LLAMA_API_URL loaded successfully.
2024-09-30 15:01:54,646 INFO __main__: HuggingFaceEmbeddings initialized successfully.
2024-09-30 15:01:55,221 INFO __main__: Chroma vector store initialized successfully.
2024-09-30 15:01:55,227 INFO __main__: Ollama LLM initialized successfully.
2024-09-30 15:01:55,228 INFO __main__: RetrievalQA chain initialized successfully.
2024-09-30 15:01:55,230 INFO __main__: Processing existing PDFs in the pdf folder.
2024-09-30 15:01:55,230 INFO __main__: Found 3 PDFs to process.
2024-09-30 15:01:55,250 INFO __main__: Existing sources in Chroma: {'FAQ - Finance for Growth course.docx.pdf', 'Finance for Growth.pdf', 'Finance for Growth coursework.docx.pdf'}
2024-09-30 15:01:55,250 INFO __main__: Skipping already processed PDF: FAQ - Finance for Growth course.docx.pdf
2024-09-30 15:01:55,250 INFO __main__: Skipping already processed PDF: Finance for Growth coursework.docx.pdf
2024-09-30 15:01:55,251 INFO __main__: Skipping already processed PDF: Finance for Growth.pdf
2024-09-30 15:01:55,251 INFO __main__: Starting Flask server on 0.0.0.0:8080
 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8080
 * Running on http://192.168.159.20:8080
Press CTRL+C to quit
127.0.0.1 - - [30/Sep/2024 15:02:05] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [30/Sep/2024 15:02:05] "GET /static/images/moon-icon.png HTTP/1.1" 200 -
127.0.0.1 - - [30/Sep/2024 15:02:05] "GET /static/images/Acharya.ai-Finance%20for%20growth.png HTTP/1.1" 200 -
127.0.0.1 - - [30/Sep/2024 15:02:05] "GET /static/css/styles.css HTTP/1.1" 200 -
127.0.0.1 - - [30/Sep/2024 15:02:05] "GET /static/js/script.js HTTP/1.1" 200 -
127.0.0.1 - - [30/Sep/2024 15:02:06] "GET /favicon.ico HTTP/1.1" 404 -
2024-09-30 15:02:26,922 INFO __main__: Received POST request at /ask_pdf endpoint.
2024-09-30 15:02:26,923 INFO __main__: Received PDF query: what are the components of the balance sheet?
E:\RAG\app.py:291: LangChainDeprecationWarning: The method `Chain.__call__` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
  response = qa_chain({"query": query})
127.0.0.1 - - [30/Sep/2024 15:02:26] "GET /static/images/corporate-user-icon.png HTTP/1.1" 200 -
2024-09-30 15:02:28,352 ERROR __main__: Failed to run QA chain: 'method' object is not iterable
127.0.0.1 - - [30/Sep/2024 15:02:28] "POST /ask_pdf HTTP/1.1" 500 -

'''