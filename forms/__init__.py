from pydantic import BaseModel

from models import Article


class CreateChatCompletionAPIForm(BaseModel):
    api_key: str
    session_id: str
    question: str
    system_prompt: str
    faiss_url: str
    pkl_url: str

class CreateEmbeddingAPIForm(BaseModel):
    api_key: str
    articles: list[Article]
