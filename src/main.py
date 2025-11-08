from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import v1_endpoints
from src.core.model_loader import model_loader

app = FastAPI(
    title="Fracture Detection API",
    description="Provides multi-level analysis of X-ray images.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_endpoints.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fracture Detection API. Go to /docs for the API info."}