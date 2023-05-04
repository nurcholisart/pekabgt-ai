from fastapi import FastAPI
from pydantic import BaseModel
from controllers import CostEstimationController

app = FastAPI()

OK = {"ok": True}


class Article(BaseModel):
    title: str
    link: str
    content: str


@app.get("/")
def root():
    return OK


@app.post("/api/v1/costs")
def cost_estimation(contents: list[str]):
    controller = CostEstimationController()
    result = controller.call(contents)

    return result


@app.post("/api/v1/embeddings")
def embedding():
    return {"faiss_url": "", "pkl_url": ""}


@app.post("/api/v1/chats")
def chat():
    return {
        "question": "",
        "answer": "",
        "articles": [{"title": "", "link": "", "content": ""}],
    }
