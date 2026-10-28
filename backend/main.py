import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import _load_env_file

# Ensure environment variables are loaded prior to router/service instantiation
_load_env_file()

from backend.api import router
from backend.database.initialize import initialize_database
from backend.routing.routes import router as routing_router
from backend.vision.router import router as vision_router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    initialize_database()
    yield


app = FastAPI(lifespan=lifespan)

# Allow local Vite dev & preview origins, plus any custom origins from .env
env_cors = os.environ.get("CORS_ALLOWED_ORIGINS", "")
custom_origins = [orig.strip() for orig in env_cors.split(",") if orig.strip()]

default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

allowed_origins = list(dict.fromkeys(default_origins + custom_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(routing_router)
app.include_router(vision_router)
