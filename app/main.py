import sys
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv

from .models.schemas import QueryModel
from .services.document_processor import extract_text_from_docx, chunk_text_nltk
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

@app.post("/upload_and_process_docx/")
async def upload_and_process_docx(file: UploadFile = File(...), index_name: Optional[str] = None):
    """
    Uploads a DOCX file, extracts text, chunks it, creates embeddings, and optionally
    stores them in Pinecone.
    """
    file_content = await file.read()
    file_name = file.filename

    print(f"Processing: {file_name}")

    text = extract_text_from_docx(file_content)
    if not text:
        raise HTTPException(status_code=400, detail="Failed to extract text from DOCX.")
    print(f"Extracted {len(text)} characters")

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

    # Store results in the global state (this state is now managed within rag_service.py)
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
                    return {"message": f"Pipeline completed successfully. Embeddings stored in Pinecone index '{index_name}'."}
                else:
                    raise HTTPException(status_code=500, detail="Failed to store embeddings in Pinecone.")
            else:
                raise HTTPException(status_code=500, detail="Failed to create/access Pinecone index.")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during Pinecone operation: {e}")
    else:
        return {"message": "Pipeline completed successfully. Embeddings will be kept in-memory."}

@app.post("/ask_document/")
async def ask_document(query_data: QueryModel):
    """
    Answers a query using the processed document content, either from in-memory or Pinecone.
    """
    if not rag_pipeline_state['embeddings_model']:
        raise HTTPException(status_code=400, detail="No document processed yet. Please upload a DOCX first.")

    llm_response = ask_llm_with_context(
        query_data.query,
        rag_pipeline_state['embeddings_model'],
        rag_pipeline_state['chunks'],
        rag_pipeline_state['embeddings'],
        pinecone_index_name=rag_pipeline_state['pinecone_index_name'],
        top_k=query_data.top_k
    )
    return {"answer": llm_response}

@app.get("/list_indexes/")
async def get_pinecone_indexes():
    """Lists available Pinecone indexes."""
    indexes = list_pinecone_indexes()
    return {"available_indexes": indexes}