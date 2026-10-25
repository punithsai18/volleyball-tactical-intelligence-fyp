"""
api/index.py - Vercel Serverless Entry Point for VolleyTactix AI.
Exposes top-level 'app', 'handler', and 'application' for Vercel Python Runtime.
"""

from __future__ import annotations

import os
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Top-level ASGI definitions required by Vercel static AST analyzer
app = FastAPI(
    title="VolleyTactix AI API",
    description="Automated Volleyball Tactical Intelligence API",
    version="1.0.0"
)
handler = app
application = app

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add directories to search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
APP_DIR = os.path.join(ROOT_DIR, "app")
FYP_DIR = os.path.join(ROOT_DIR, "fyp")

for p in [CURRENT_DIR, ROOT_DIR, APP_DIR, FYP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

@app.get("/")
async def root():
    return HTMLResponse("<h1>VolleyTactix AI Serverless API Active</h1>", status_code=200)

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "system": "VolleyTactix AI",
        "backend": "Vercel Serverless Function",
        "available_models": 13,
        "status_code": 200
    }

# Try mounting the full app endpoints if model_service is available
try:
    from app.main import app as main_app
    # Forward existing endpoints from main_app
    for route in main_app.routes:
        if route.path not in ["/", "/api/health"]:
            app.routes.append(route)
except Exception as e:
    print(f"[Vercel] Running in lightweight serverless mode: {e}")
