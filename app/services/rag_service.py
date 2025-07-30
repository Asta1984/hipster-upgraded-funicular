import requests
import json 
import os 
from .pinecone_service import search_similar_chunks
from dotenv import load_dotenv

load_dotenv()

rag_pipeline_state = {
    'chunks': [],
    'embeddings': [],
    'metadata': [],
    'embeddings_model': None,
    'pinecone_index_name': None
}

def call_ollama_llm(prompt, ollama_url=os.getenv('OLLAMA_BASE_URL'), model="tinyllama:latest"):
    api_endpoint = f"{ollama_url}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    print(f"Calling Ollama LLM ({model})...")
    full_response_content = ""

    try:
        with requests.post(api_endpoint, headers=headers, json=data, stream=True, timeout=300) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        json_data = json.loads(line.decode('utf-8'))
                        if "response" in json_data:
                            full_response_content += json_data["response"]
                        elif "content" in json_data.get("message", {}):
                             full_response_content += json_data["message"]["content"]

                        if json_data.get("done"):
                            break
                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError on line: {line.decode('utf-8')}, Error: {e}")
                        continue

            if full_response_content:
                print("\nOllama LLM response received.")
                return full_response_content
            else:
                return f"Ollama LLM response received, but no content was extracted."

    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error calling Ollama LLM: {http_err} - Response: {response.text}"
    except requests.exceptions.ConnectionError as conn_err:
        return f"Connection error calling Ollama LLM: {conn_err}. Make sure Ollama server is running and accessible."
    except requests.exceptions.Timeout as timeout_err:
        return f"Timeout error calling Ollama LLM: {timeout_err}. The request took too long."
    except requests.exceptions.RequestException as req_err:
        return f"An unexpected request error occurred calling Ollama LLM: {req_err}"
    except Exception as e:
        return f"An unexpected error occurred during Ollama LLM call: {e}"

def ask_llm_with_context(query, embeddings_model, chunks, embeddings_list, pinecone_index_name=None, top_k=5):
    """
    Searches for relevant chunks and uses an LLM to answer the query based on context.
    """
    print(f"\n--- Answering query with LLM ---")
    print(f"User Query: '{query}'")

    retrieved_chunks_info = search_similar_chunks(
        query,
        embeddings_model,
        chunks,
        embeddings_list,
        top_k=top_k,
        pinecone_index_name=pinecone_index_name
    )

    if not retrieved_chunks_info:
        return "No relevant information found in the document to answer your query. Please try a different question."

    context_texts = [info[2] for info in retrieved_chunks_info]
    context = "\n\n".join(context_texts)

    llm_prompt = f"""
    You are an AI assistant tasked with answering questions based on the provided context.

Context:
---
{context}
---

Question: {query}

Answer:
"""
    #print("\n--- Sending prompt to LLM ---")
    #print(f"Prompt preview:\n{llm_prompt[:50]}...\n[Full prompt length: {len(llm_prompt)} characters]")

    llm_response = call_ollama_llm(llm_prompt)

    print("\n--- LLM's Final Answer ---")
    print(llm_response)
    print("--------------------------")
    return llm_response