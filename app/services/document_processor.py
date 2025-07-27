import nltk
from langchain_text_splitters import NLTKTextSplitter
from docx import Document
from io import BytesIO
import os

def setup_nltk_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))  
    nltk_data_path = os.path.join(project_root, 'nltk_data')
    print(f"Looking for NLTK data at: {nltk_data_path}")
    
    if os.path.exists(nltk_data_path):
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_path)
        print(f"Added {nltk_data_path} to NLTK data path")
    else:
        print(f"Local NLTK data not found at {nltk_data_path}")

setup_nltk_data()

try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK punkt tokenizer found!")
except LookupError:
    print("Punkt tokenizer not found locally, attempting download...")
    try:
        nltk.download('punkt', quiet=True)
        print("NLTK punkt download complete.")
    except Exception as e:
        print(f"Failed to download punkt: {e}")
        raise

def extract_text_from_docx(file_content: bytes):
    """Extract text from DOCX file from bytes content"""
    try:
        doc = Document(BytesIO(file_content))
        paragraphs = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                paragraphs.append(text)
        return '\n\n'.join(paragraphs)
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return None

def chunk_text_nltk(text, chunk_size=200, chunk_overlap=20):
    """Chunk text using NLTK splitter"""
    splitter = NLTKTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)