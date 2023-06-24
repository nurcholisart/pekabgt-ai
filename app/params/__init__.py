from app.models import Article, Content

from pydantic import BaseModel


class CreateTokenCalculationAPIParam(BaseModel):
    model: str = "gpt-4"
    contents: list[str]


class CreateChatCompletionAPIParam(BaseModel):
    api_key: str
    session_id: str
    question: str
    system_prompt: str
    faiss_url: str
    pkl_url: str


class CreateEmbeddingAPIParam(BaseModel):
    api_key: str
    contents: list[Content]
