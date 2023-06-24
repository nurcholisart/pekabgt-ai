from fastapi import FastAPI
from config.routes import base_router

peka_bgt_ai_app = FastAPI(
    title="PekaBGT AI",
    description="A PekaBGT's service to communicate with OpenAI",
)


peka_bgt_ai_app.include_router(base_router)
