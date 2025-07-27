from pydantic import BaseModel
from typing import List, Optional


class QueryModel(BaseModel):
    query: str
    top_k: Optional[int] = 5
    pinecone_index_name: Optional[str] = None
