from fastapi import FastAPI
from pydantic import BaseModel
from controllers import CostEstimationController, EmbeddingController, ChatController
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
    chatbot_name: str
    chatbot_description: str
    faiss_url: str
    pkl_url: str
    chat_history: list[tuple] = []


@app.post("/api/v1/{app_code}/chats")
def chat(app_code: str, body: ChatRequest):
    controller = ChatController()
    result = controller.call(
        api_key=body.api_key,
        faiss_url=body.faiss_url,
        pkl_url=body.pkl_url,
        chat_history=body.chat_history,
        question=body.question,
        chatbot_name=body.chatbot_name,
        chatbot_description=body.chatbot_description
    )

    return result
