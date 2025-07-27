import os
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings 
from fastapi import HTTPException 

def setup_pinecone_api():
    """Setup Pinecone API key securely"""
    if 'PINECONE_API_KEY' not in os.environ:
        raise RuntimeError("PINECONE_API_KEY environment variable not set.")
    print("Pinecone API key configured!")

def initialize_embeddings(model_name='multilingual-e5-large'):
    """Initialize Pinecone embeddings"""
    try:
        embeddings = PineconeEmbeddings(
            model=model_name,
            pinecone_api_key=os.environ.get('PINECONE_API_KEY')
        )
        print(f"Embeddings initialized with model: {model_name}")
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        return None

def create_embeddings_from_chunks(chunks, embeddings_model):
    """Create embeddings for text chunks"""
    print(f"🔄 Creating embeddings for {len(chunks)} chunks...")
    try:
        embeddings_list = embeddings_model.embed_documents(chunks)
        print(f"Created {len(embeddings_list)} embeddings")
        if embeddings_list:
            print(f"Embedding dimension: {len(embeddings_list[0])}")
        return embeddings_list
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return None

def create_embedding_metadata(chunks, source_file=None):
    """Create metadata for each chunk"""
    metadata_list = []
    for i, chunk in enumerate(chunks):
        metadata = {
            'chunk_id': i,
            'chunk_size': len(chunk),
            'source_file': source_file or 'uploaded_document',
            'chunk_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk,
            'text': chunk
        }
        metadata_list.append(metadata)
    return metadata_list

# === Pinecone index operations (No change to logic) ===
def create_pinecone_index(index_name, dimension, cloud='aws', region='us-east-1'):
    """Create a new Pinecone index if it doesn't exist"""
    try:
        pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        existing_indexes = [idx['name'] for idx in pc.list_indexes()]
        if index_name in existing_indexes:
            print(f"Index '{index_name}' already exists")
            return True
        spec = ServerlessSpec(cloud=cloud, region=region)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=spec
        )
        print(f"Created new index '{index_name}' with dimension {dimension}")
        return True
    except Exception as e:
        print(f"Error creating index: {e}")
        raise HTTPException(status_code=500, detail=f"Pinecone index creation error: {e}")

def list_pinecone_indexes():
    """List available Pinecone indexes"""
    try:
        pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        indexes = pc.list_indexes()
        if indexes:
            print("Available Pinecone indexes:")
            for idx in indexes:
                print(f"  - {idx['name']} (dimension: {idx['dimension']})")
        else:
            print("No Pinecone indexes found")
        return [idx['name'] for idx in indexes]
    except Exception as e:
        print(f"Error listing indexes: {e}")
        return []

def store_embeddings_in_pinecone(embeddings_list, chunks, metadata_list, index_name):
    """Store embeddings in Pinecone index"""
    try:
        pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        index = pc.Index(index_name)
        vectors = []
        for i, (embedding, chunk, metadata) in enumerate(zip(embeddings_list, chunks, metadata_list)):
            vector_id = f"chunk_{i}_{hash(chunk[:50]) % 10000}"
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    **metadata,
                    'text': chunk
                }
            })
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
        print(f"Successfully stored {len(vectors)} embeddings in Pinecone index '{index_name}'")
        stats = index.describe_index_stats()
        print(f"Index stats: {stats['total_vector_count']} total vectors")
        return True
    except Exception as e:
        print(f"Error storing embeddings in Pinecone: {e}")
        return False

# === Utility functions for working with embeddings (No change to logic) ===
def search_similar_chunks(query, embeddings_model, chunks, embeddings_list, top_k=3, pinecone_index_name=None):
    """Search for similar chunks using cosine similarity from in-memory or Pinecone"""
    query_embedding = embeddings_model.embed_query(query)

    if pinecone_index_name:
        try:
            pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
            index = pc.Index(pinecone_index_name)
            query_results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            similarities = []
            for match in query_results.matches:
                similarities.append((match.id, match.score, match.metadata['text']))
            print(f"🔍 Top {top_k} similar chunks from Pinecone for query: '{query}'")
            return similarities
        except Exception as e:
            print(f"Error searching Pinecone: {e}. Falling back to in-memory search if data available.")
            if chunks and embeddings_list:
                return _in_memory_search(query, query_embedding, chunks, embeddings_list, top_k)
            else:
                return []
    else:
        return _in_memory_search(query, query_embedding, chunks, embeddings_list, top_k)

def _in_memory_search(query, query_embedding, chunks, embeddings_list, top_k):
    """Helper for in-memory search"""
    similarities = []
    for i, chunk_embedding in enumerate(embeddings_list):
        dot_product = np.dot(query_embedding, chunk_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_chunk = np.linalg.norm(chunk_embedding)

        if norm_query == 0 or norm_chunk == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm_query * norm_chunk)
        similarities.append((i, similarity, chunks[i]))

    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"🔍 Top {top_k} similar chunks (in-memory) for query: '{query}'")
    return similarities[:top_k]