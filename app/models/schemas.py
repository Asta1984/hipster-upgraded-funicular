from pydantic import BaseModel
from typing import List, Optional

class QueryModel(BaseModel):
    query: str
    top_k: int = 5

