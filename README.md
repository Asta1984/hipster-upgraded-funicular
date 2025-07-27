# RAG Chatbot System

A Retrieval-Augmented Generation (RAG) chatbot system built with FastAPI backend and Streamlit frontend, designed to answer questions based on uploaded DOCX documents using Pinecone vector database and Ollama LLM.

## 🏗️ Architecture

The system consists of three main components:

1. **FastAPI Backend** (`app/main.py`) - Document processing and RAG pipeline API
2. **Streamlit Frontend** (`app.py`) - User interface for chat interactions
3. **External Services** - Pinecone (vector database) and Ollama (local LLM)

## 📋 Features

- **Document Upload**: Process DOCX files and extract text content
- **Text Chunking**: Intelligent text splitting using NLTK
- **Vector Embeddings**: Generate embeddings using multilingual-e5-large model
- **Vector Storage**: Store embeddings in Pinecone or keep in-memory
- **Semantic Search**: Find relevant document chunks based on user queries
- **LLM Integration**: Generate answers using Ollama local LLM
- **Chat Interface**: User-friendly Streamlit chat interface
- **Real-time Streaming**: Simulated streaming responses for better UX

## 🛠️ Prerequisites

### Required Services

1. **Pinecone Account**
   - Sign up at [Pinecone](https://www.pinecone.io/)
   - Get your API key from the dashboard

2. **Ollama Installation**
   - Install Ollama from [ollama.ai](https://ollama.ai/)
   - Pull a model (e.g., `ollama pull tinyllama:latest`)
   - Start Ollama server: `ollama serve`

### System Requirements

- Python 3.8+
- Internet connection for Pinecone API
- Local Ollama server running

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

The application will automatically download NLTK punkt tokenizer on first run, but you can pre-download it:

```python
import nltk
nltk.download('punkt')
```

## ⚙️ Configuration

### 1. Environment Variables

Create a `.env` file in the root directory using `.env.example` as template:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment
DEFAULT_INDEX_NAME=rag-chatbot-index

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=tinyllama:latest

# Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE=10485760

# API Configuration
DEBUG=false
RAG_BACKEND_URL=http://localhost:8000/ask_document/
```

### 2. Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `PINECONE_API_KEY` | Your Pinecone API key | Required |
| `PINECONE_ENVIRONMENT` | Pinecone environment | Required |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `DEFAULT_MODEL` | Ollama model to use | `tinyllama:latest` |
| `CHUNK_SIZE` | Text chunk size for processing | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `MAX_FILE_SIZE` | Maximum file size in bytes | `10485760` (10MB) |

## 🏃‍♂️ Usage

### 1. Start the FastAPI Backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 2. Start the Streamlit Frontend

In a new terminal window:

```bash
streamlit run app.py
```

The chat interface will be available at `http://localhost:8501`

### 3. Using the System

#### Upload and Process Documents

1. **Via API** (recommended for initial setup):
   ```bash
   curl -X POST "http://localhost:8000/upload_and_process_docx/" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@your_document.docx" \
        -F "index_name=my-rag-index"
   ```

2. **Via Streamlit**: Use the chat interface to upload documents (feature may need to be added)

#### Ask Questions

1. **Via Streamlit**: Type your questions in the chat interface
2. **Via API**:
   ```bash
   curl -X POST "http://localhost:8000/ask_document/" \
        -H "Content-Type: application/json" \
        -d '{"query": "What is the main topic of the document?", "top_k": 5}'
   ```

## 📚 API Documentation

### Endpoints

#### `POST /upload_and_process_docx/`
Upload and process a DOCX file.

**Parameters:**
- `file`: DOCX file (multipart/form-data)
- `index_name`: Optional Pinecone index name

**Response:**
```json
{
  "message": "Pipeline completed successfully. Embeddings stored in Pinecone index 'my-index'."
}
```

#### `POST /ask_document/`
Ask a question about the processed document.

**Request Body:**
```json
{
  "query": "Your question here",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Generated answer based on document content"
}
```

#### `GET /list_indexes/`
List available Pinecone indexes.

**Response:**
```json
{
  "available_indexes": ["index1", "index2"]
}
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI documentation.

## 🔧 Troubleshooting

### Common Issues

1. **Pinecone Connection Error**
   ```
   Error: Could not connect to Pinecone
   ```
   - Verify your `PINECONE_API_KEY` is correct
   - Check your internet connection
   - Ensure Pinecone service is not down

2. **Ollama Connection Error**
   ```
   Connection error calling Ollama LLM
   ```
   - Make sure Ollama is installed and running: `ollama serve`
   - Verify the `OLLAMA_BASE_URL` is correct
   - Check if the specified model is available: `ollama list`

3. **NLTK Download Issues**
   ```
   LookupError: punkt tokenizer not found
   ```
   - The app should auto-download, but manually run:
   ```python
   import nltk
   nltk.download('punkt')
   ```

4. **File Upload Issues**
   - Ensure file is a valid DOCX format
   - Check file size is under the limit (default 10MB)
   - Verify file is not corrupted

### Performance Tips

1. **Chunk Size Optimization**
   - Smaller chunks (200-500): Better for specific questions
   - Larger chunks (1000-2000): Better for context-heavy questions

2. **Embedding Model Selection**
   - `multilingual-e5-large`: Good for multilingual content
   - Consider other models based on your content language

3. **Pinecone vs In-Memory**
   - Use Pinecone for persistent storage and large documents
   - Use in-memory for testing and small documents

## 🔒 Security Considerations

- Store API keys securely using environment variables
- Never commit `.env` files to version control
- Consider implementing authentication for production use
- Validate file uploads to prevent malicious content

## 🚀 Deployment

### Docker Deployment (Recommended)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t rag-chatbot .
docker run -p 8000:8000 --env-file .env rag-chatbot
```

### Production Considerations

1. Use a production WSGI server (e.g., Gunicorn)
2. Set up proper logging and monitoring
3. Implement rate limiting
4. Use HTTPS in production
5. Set up backup strategies for Pinecone data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review API documentation at `/docs`
3. Create an issue in the repository
4. Check Pinecone and Ollama documentation for service-specific issues

## 🔄 Version History

- **v1.0.0**: Initial release with basic RAG functionality
- Document upload and processing
- Pinecone integration
- Ollama LLM integration
- Streamlit chat interface