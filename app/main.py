import sys
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv

from .models.schemas import QueryModel
from .services.document_processor import extract_text_from_docx, extract_text_from_markdown, chunk_text_nltk
from .services.pinecone_service import (
    setup_pinecone_api,
    initialize_embeddings,
    create_embeddings_from_chunks,
    create_embedding_metadata,
    create_pinecone_index,
    store_embeddings_in_pinecone,
    list_pinecone_indexes
)
from .services.rag_service import ask_llm_with_context, rag_pipeline_state


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI application starting up...")
    try:
        setup_pinecone_api() 
    except RuntimeError as e:
        print(f"Startup failed: {e}")
        sys.exit(1)
    yield
  
    print("FastAPI application shutting down...")

app = FastAPI(lifespan=lifespan)

@app.post("/upload_and_process_document/")
async def upload_and_process_document(file: UploadFile = File(...), index_name: Optional[str] = None):
    """
    Uploads a document file (DOCX or Markdown), extracts text, chunks it, creates embeddings, 
    and optionally stores them in Pinecone.
    """
    file_content = await file.read()
    file_name = file.filename
    
    print(f"Processing: {file_name}")
    
    # Determine file type and extract text accordingly
    file_extension = file_name.lower().split('.')[-1] if '.' in file_name else ''
    
    if file_extension in ['docx']:
        text = extract_text_from_docx(file_content)
        if not text:
            raise HTTPException(status_code=400, detail="Failed to extract text from DOCX file.")
    elif file_extension in ['md', 'markdown']:
        text = extract_text_from_markdown(file_content)
        if not text:
            raise HTTPException(status_code=400, detail="Failed to extract text from Markdown file.")
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {file_extension}. Supported formats: DOCX, Markdown (.md, .markdown)"
        )
    
    print(f"Extracted {len(text)} characters from {file_extension.upper()} file")

    CHUNK_SIZE = 200
    CHUNK_OVERLAP = 20
    chunks = chunk_text_nltk(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks")

    embeddings_model = initialize_embeddings('multilingual-e5-large')
    if not embeddings_model:
        raise HTTPException(status_code=500, detail="Failed to initialize embeddings model.")

    embeddings_list = create_embeddings_from_chunks(chunks, embeddings_model)
    if not embeddings_list:
        raise HTTPException(status_code=500, detail="Failed to create embeddings.")

    metadata_list = create_embedding_metadata(chunks, file_name)

    # Store results in the global state
    rag_pipeline_state['chunks'] = chunks
    rag_pipeline_state['embeddings'] = embeddings_list
    rag_pipeline_state['metadata'] = metadata_list
    rag_pipeline_state['embeddings_model'] = embeddings_model

    if index_name:
        embedding_dimension = len(embeddings_list[0])
        try:
            create_success = create_pinecone_index(index_name, embedding_dimension)
            if create_success:
                success = store_embeddings_in_pinecone(embeddings_list, chunks, metadata_list, index_name)
                if success:
                    rag_pipeline_state['pinecone_index_name'] = index_name
                    return {
                        "message": f"Pipeline completed successfully. {file_extension.upper()} file processed and embeddings stored in Pinecone index '{index_name}'.",
                        "file_type": file_extension,
                        "chunks_created": len(chunks),
                        "characters_extracted": len(text)
                    }
                else:
                    raise HTTPException(status_code=500, detail="Failed to store embeddings in Pinecone.")
            else:
                raise HTTPException(status_code=500, detail="Failed to create/access Pinecone index.")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during Pinecone operation: {e}")
    else:
        return {
            "message": f"Pipeline completed successfully. {file_extension.upper()} file processed and embeddings kept in-memory.",
            "file_type": file_extension,
            "chunks_created": len(chunks),
            "characters_extracted": len(text)
        }

@app.post("/ask_document/")
async def ask_document(query_data: QueryModel):
    """
    Answers a query using either in-memory document content or a Pinecone index.
    """
    # Initialize model if not yet done
    if not rag_pipeline_state['embeddings_model']:
        embeddings_model = initialize_embeddings('multilingual-e5-large')
        if not embeddings_model:
            raise HTTPException(status_code=500, detail="Failed to initialize embeddings model.")
        rag_pipeline_state['embeddings_model'] = embeddings_model

    # Use the index if provided
    pinecone_index = rag_pipeline_state.get('pinecone_index_name') or query_data.pinecone_index_name
    if not pinecone_index and not rag_pipeline_state.get('chunks'):
        raise HTTPException(status_code=400, detail="No document in memory or Pinecone index provided.")

    llm_response = ask_llm_with_context(
        query_data.query,
        rag_pipeline_state['embeddings_model'],
        rag_pipeline_state.get('chunks'),
        rag_pipeline_state.get('embeddings'),
        pinecone_index_name=pinecone_index,
        top_k=query_data.top_k
    )
    return {"answer": llm_response}

@app.get("/list_indexes/")
async def get_pinecone_indexes():
    """Lists available Pinecone indexes."""
    indexes = list_pinecone_indexes()
    return {"available_indexes": indexes}