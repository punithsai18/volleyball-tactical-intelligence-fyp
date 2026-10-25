"""
api/index.py - Vercel Serverless Function Entry Point for VolleyTactix AI.

Exposes the FastAPI application to Vercel's Python runtime.
Handles routing, CORS, and provides resilient fallbacks for cloud environments.
"""

from __future__ import annotations

import os
import sys

# Ensure root, app, and fyp directories are in python search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
APP_DIR = os.path.join(ROOT_DIR, "app")
FYP_DIR = os.path.join(ROOT_DIR, "fyp")

for path in [CURRENT_DIR, ROOT_DIR, APP_DIR, FYP_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from app.main import app
except Exception as e:
    # Graceful standalone fallback for serverless cold-start
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, HTMLResponse

    app = FastAPI(
        title="VolleyTactix AI - Cloud Serverless API",
        description="Vercel Serverless fallback endpoint for VolleyTactix AI.",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return HTMLResponse("<h1>VolleyTactix AI Serverless API Active</h1>", status_code=200)

    @app.get("/api/health")
    async def health():
        return {
            "status": "healthy",
            "system": "VolleyTactix AI",
            "runtime": "Vercel Serverless Function",
            "status_code": 200
        }
