from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="MediBot AI")

app.include_router(router)