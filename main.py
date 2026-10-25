"""
main.py - Root Entrypoint for VolleyTactix AI on Vercel.
Serves the complete frontend suite and connects to the Render ML backend.
"""

from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

# Top-level ASGI app recognized by Vercel
app = FastAPI(
    title="VolleyTactix AI",
    description="Automated Volleyball Tactical Intelligence Dashboard",
    version="1.0.0"
)
handler = app
application = app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT_DIR, "app", "static")

# Mount /static route
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>VolleyTactix AI Dashboard</h1>")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "system": "VolleyTactix AI Website",
        "backend_target": "Render ML Engine",
        "status_code": 200
    }

@app.get("/{full_path:path}")
async def serve_file(full_path: str):
    file_path = os.path.join(STATIC_DIR, full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return HTMLResponse("File Not Found", status_code=404)
