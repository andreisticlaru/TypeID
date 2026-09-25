"""FastAPI application entrypoint.

Run with: `uvicorn app.main:app --reload` from within `backend/`.
"""

import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

# `features/` lives at the repo root, one level above `backend/`. Make it
# importable so this app can use the single canonical feature-extraction
# implementation (per CLAUDE.md) rather than a duplicate.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .gallery import init_db
from .routers import authenticate, enroll, identify, map as map_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # UMAP's first fit compiles for ~20 s; do it now so the first /map visit is instant.
    threading.Thread(target=map_router.map_points, daemon=True).start()
    yield


app = FastAPI(title="TypeID backend", lifespan=lifespan)

# Vite dev server default port -- a teammate is scaffolding a Vite frontend
# in parallel to the existing plain-JS one.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://frlnfv6t-5173.euw.devtunnels.ms",  # dev tunnel forwarding the frontend port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(authenticate.router)
app.include_router(enroll.router)
app.include_router(identify.router)
app.include_router(map_router.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
