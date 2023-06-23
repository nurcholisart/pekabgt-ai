from fastapi import FastAPI
from pydantic import BaseModel
from controllers import (
    CostEstimationController,
    EmbeddingController,
    ChatController,
    V2ChatController,
)
from models import Article

app = FastAPI()

OK = {"ok": True}


@app.get("/")
def root():
    return OK


@app.post("/api/v1/{app_code}/costs")
def cost_estimation(contents: list[str]):
    controller = CostEstimationController()
    result = controller.call(contents)

    return result


class EmbeddingRequest(BaseModel):
    api_key: str
    articles: list[Article]


@app.post("/api/v1/{app_code}/embeddings")
def embedding(app_code: str, body: EmbeddingRequest):
    controller = EmbeddingController()
    result = controller.call(body.api_key, app_code, body.articles)

    return result


class ChatRequest(BaseModel):
    question: str
    api_key: str
    system_prompt: str
    faiss_url: str
    pkl_url: str
    chat_history: list[tuple] = []


@app.post("/api/v1/{app_code}/chats")
def chat(app_code: str, body: ChatRequest):
    print(body.json())
    controller = ChatController()
    result = controller.call(
        api_key=body.api_key,
        faiss_url=body.faiss_url,
        pkl_url=body.pkl_url,
        chat_history=body.chat_history,
        question=body.question,
        system_prompt=body.system_prompt,
    )

    return result


class V2ChatRequest(BaseModel):
    api_key: str
    session_id: str
    question: str
    system_prompt: str
    faiss_url: str
    pkl_url: str


@app.post("/api/v2/{app_code}/chats")
def chat_v2(app_code: str, body: V2ChatRequest):
    controller = V2ChatController(
        body.api_key,
        body.session_id,
        body.question,
        body.system_prompt,
        body.faiss_url,
        body.pkl_url,
    )

    return controller.call()
