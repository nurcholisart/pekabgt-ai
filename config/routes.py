from typing import Literal
from fastapi import APIRouter

from app.responses import (
    RootResponse,
    CreateTokenCalculationAPIResponse,
    CreateEmbeddingAPIResponse,
    CreateChatCompletionAPIResponse,
)

from app.params import CreateTokenCalculationAPIParam

from app.controllers import CreateTokenCalculationController

base_router = APIRouter()


@base_router.get("/", description="Root path", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(ok=True)


@base_router.post(
    "/api/v2/token-calculations",
    description="Calculate number of tokens and its estimation cost per 1000 tokens based on given text",
    response_model=CreateTokenCalculationAPIResponse,
)
def create_token_calculations(params: CreateTokenCalculationAPIParam) -> RootResponse:
    controller = CreateTokenCalculationController(params)

    return controller()


@base_router.post(
    "/api/v2/embeddings",
    description="Create embedding based on given text",
    response_model=CreateEmbeddingAPIResponse,
)
def create_embeddings() -> RootResponse:
    return RootResponse(ok=True)


@base_router.post(
    "/api/v2/chat-completions",
    description="Create chat completion",
    response_model=CreateChatCompletionAPIResponse,
)
def create_chat_completions() -> RootResponse:
    return RootResponse(ok=True)
