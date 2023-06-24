from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.params import CreateChatCompletionAPIParam, CreateEmbeddingAPIParam

from controllers import (
    CostEstimationController,
    EmbeddingController,
    V2ChatController,
)

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

@app.post("/api/v1/{app_code}/embeddings")
def embedding(app_code: str, body: CreateChatCompletionAPIParam):
    controller = EmbeddingController()
    result = controller.call(body.api_key, app_code, body.articles)

    return result


@app.post("/api/v2/{app_code}/chats")
def chat_v2(app_code: str, body: CreateEmbeddingAPIParam) -> JSONResponse:
    controller = V2ChatController(
        body.api_key,
        body.session_id,
        body.question,
        body.system_prompt,
        body.faiss_url,
        body.pkl_url,
    )

    return controller.call()