from pydantic import BaseModel
from .article import Article

class Content(BaseModel):
    title: str
    metadata: dict