from pydantic import BaseModel


class RootResponse(BaseModel):
    ok: bool


class CreateTokenCalculationAPIResponse(BaseModel):
    words_count: int
    tokens_count: int
    estimated_cost: str
    currency: str = "USD"


class CreateChatCompletionAPIResponse(BaseModel):
    ok: bool


class CreateEmbeddingAPIResponse(BaseModel):
    ok: bool
