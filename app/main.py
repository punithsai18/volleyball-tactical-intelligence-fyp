"""
main.py - FastAPI Application for Volleyball Tactical Intelligence.

Serves the modern analytics dashboard and exposes REST endpoints for
tactical formation inference, model benchmarking, and Olympic match telemetry.
"""

from __future__ import annotations

import io
import os
import sys
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add current dir & repo root to path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import model_service

app = FastAPI(
    title="VolleyTactix AI - Automated Tactical Intelligence",
    description="Whole-team spatial coordination analytics from broadcast video using dual detection, formation tracking, and 13 learned sequence classifiers.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = os.path.join(APP_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictRequest(BaseModel):
    model_key: str = "hist_gradient_boosting"
    sample_id: Optional[str] = None
    # 29 rows x 14 cols or list of frame coordinate dicts
    coordinates: Optional[List[List[float]]] = None


class CustomFormationRequest(BaseModel):
    model_key: str = "hist_gradient_boosting"
    # Single frame: ball [x, y], players [[x, y], ... 6 players]
    ball: List[float] # [x, y]
    players: List[List[float]] # 6 x [x, y]


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>VolleyTactix AI Dashboard Loading...</h1>", status_code=200)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "system": "VolleyTactix AI",
        "backend": "FastAPI + PyTorch + Scikit-Learn 1.5.1",
        "available_models": len(model_service.MODEL_REGISTRY),
        "cached_models": list(model_service._LOADED_MODELS.keys())
    }


@app.get("/api/models")
async def get_models():
    """Returns the list of 13 models with metadata and leaderboard statistics."""
    models_list = []
    for key, val in model_service.MODEL_REGISTRY.items():
        models_list.append({
            "key": key,
            "name": val["name"],
            "type": val["type"],
            "rank": val["rank"],
            "test_acc": val["test_acc"],
            "test_f1": val["test_f1"],
            "train_acc": val["train_acc"],
            "description": val["description"],
            "badge": val["badge"]
        })
    models_list.sort(key=lambda x: x["rank"])
    
    leaderboard = model_service.get_leaderboard_data()
    return {
        "models": models_list,
        "leaderboard": leaderboard
    }


@app.get("/api/samples")
async def get_samples():
    """Returns curated rally samples across all 4 tactical classes."""
    samples = model_service.get_available_samples()
    return {"samples": samples}


@app.get("/api/sample/{sample_id}")
async def get_sample_prediction(sample_id: str, model_key: str = "hist_gradient_boosting"):
    """Load a sample rally clip and run tactical prediction."""
    try:
        raw = model_service.load_sample_clip_data(sample_id)
        result = model_service.predict_from_raw_array(raw, model_key=model_key, include_frames=True)
        result["sample_id"] = sample_id
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict")
async def predict_coordinates(req: PredictRequest):
    """
    Run tactical prediction from either a sample ID or raw 29x14 coordinates.
    """
    try:
        if req.sample_id:
            raw = model_service.load_sample_clip_data(req.sample_id)
        elif req.coordinates:
            raw = np.array(req.coordinates, dtype=float)
            if raw.shape[1] != 14:
                raise ValueError(f"Expected 14 columns, got {raw.shape[1]}")
        else:
            raise ValueError("Either 'sample_id' or 'coordinates' must be provided.")
        
        result = model_service.predict_from_raw_array(raw, model_key=req.model_key, include_frames=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict-all")
async def predict_all_consensus(req: PredictRequest):
    """
    Run prediction across ALL 13 models simultaneously to assess consensus and disagreement.
    """
    try:
        if req.sample_id:
            raw = model_service.load_sample_clip_data(req.sample_id)
        elif req.coordinates:
            raw = np.array(req.coordinates, dtype=float)
        else:
            raise ValueError("Either 'sample_id' or 'coordinates' must be provided.")
        
        consensus_data = model_service.predict_all_models(raw)
        return consensus_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict-custom-formation")
async def predict_custom_formation(req: CustomFormationRequest):
    """
    Synthesize a 29-frame smooth formation sequence from user-placed players and ball,
    and predict the tactical category in real-time!
    """
    try:
        if len(req.players) != 6:
            raise ValueError("Exactly 6 players required.")
        
        # Build 29 repeated / smoothly jiggling frames for steady-state evaluation
        frames = []
        base_ball = np.array(req.ball, dtype=float)
        base_players = np.array(req.players, dtype=float)
        
        for t in range(29):
            # Subtle realistic kinetic micro-oscillation (0.2px)
            noise_p = np.sin(t * 0.2) * 0.5
            noise_b = np.cos(t * 0.3) * 0.5
            
            row = [
                base_ball[0] + noise_b,
                base_ball[1] + noise_b
            ]
            for p_idx in range(6):
                row.extend([
                    base_players[p_idx][0] + noise_p,
                    base_players[p_idx][1] + noise_p
                ])
            frames.append(row)
            
        raw = np.array(frames, dtype=float)
        result = model_service.predict_from_raw_array(raw, model_key=req.model_key, include_frames=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/upload-csv")
async def upload_csv_file(file: UploadFile = File(...), model_key: str = Form("hist_gradient_boosting")):
    """
    Upload a rally CSV file with columns (frame_id, ball_x, ball_y, p1_x, p1_y, ..., p6_x, p6_y)
    and receive full tactical analysis.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        
        missing = [c for c in model_service.RAW_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Uploaded CSV is missing required columns: {missing}")
        
        raw = df[model_service.RAW_COLS].to_numpy()
        result = model_service.predict_from_raw_array(raw, model_key=model_key, include_frames=True)
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/sample-videos")
async def get_sample_videos():
    """Return available volleyball match videos ready for 1-click test analysis."""
    sample_list = []
    
    # Check pre-rendered demo
    pre_rendered = os.path.join(APP_DIR, "static", "outputs", "coord_attack_demo.mp4")
    if os.path.exists(pre_rendered):
        sample_list.append({
            "id": "coord_attack_demo",
            "name": "Japan vs Argentina Rally (Attack Sequence)",
            "path": pre_rendered,
            "is_precomputed": True,
            "url": "/static/outputs/coord_attack_demo.mp4",
            "dominant_tactic": "Coordinated Attack",
            "duration": "1.4s (43 frames)"
        })
        
    # Check AdobeEditing sample
    raw_sample = r"D:\AdobeEditing\coordAttack.mp4"
    if os.path.exists(raw_sample):
        sample_list.append({
            "id": "raw_coord_attack",
            "name": "Broadcast Match Clip (coordAttack.mp4)",
            "path": raw_sample,
            "is_precomputed": False,
            "dominant_tactic": "Coordinated Attack",
            "duration": "17.6 MB Raw Video"
        })

    return {"sample_videos": sample_list}


@app.post("/api/upload-video")
async def upload_video_file(
    file: UploadFile = File(...),
    model_key: str = Form("hist_gradient_boosting"),
    max_duration_sec: float = Form(8.0)
):
    """
    Upload a match video (MP4/MOV/AVI), run player/ball detection + 18-dim formation tracking,
    execute tactical sequence classifier, and return rendered annotated video + timeline analytics.
    """
    import uuid
    import video_service
    
    task_id = str(uuid.uuid4())[:8]
    temp_upload_dir = os.path.join(APP_DIR, "static", "outputs", "uploads")
    os.makedirs(temp_upload_dir, exist_ok=True)
    
    temp_in_path = os.path.join(temp_upload_dir, f"{task_id}_{file.filename}")
    out_filename = f"tactical_analysis_{task_id}.mp4"
    
    try:
        # Save uploaded file
        with open(temp_in_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
                
        # Run video processing
        result = video_service.process_video_file(
            input_video_path=temp_in_path,
            output_filename=out_filename,
            model_key=model_key,
            max_duration_sec=float(max_duration_sec)
        )
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up temporary upload file
        if os.path.exists(temp_in_path):
            try:
                os.remove(temp_in_path)
            except Exception:
                pass


@app.post("/api/process-sample-video")
async def process_sample_video(
    sample_id: str = Form("raw_coord_attack"),
    model_key: str = Form("hist_gradient_boosting"),
    max_duration_sec: float = Form(5.0)
):
    """Analyze one of the pre-loaded local sample videos directly."""
    import video_service
    import uuid

    if sample_id == "coord_attack_demo":
        # Return already rendered video
        return {
            "output_video_url": "/static/outputs/coord_attack_demo.mp4",
            "output_filename": "coord_attack_demo.mp4",
            "processed_frames": 43,
            "duration_sec": 1.43,
            "fps": 30.0,
            "processing_time_sec": 0.05,
            "dominant_tactic": "Coordinated Attack",
            "dominant_color": "#f59e0b",
            "model_used": "HistGradientBoosting",
            "windows": [
                {
                    "second": 0.9,
                    "window_idx": 1,
                    "tactic": "Coordinated Attack",
                    "confidence": 0.814,
                    "entropy": 0.512,
                    "color": "#f59e0b"
                },
                {
                    "second": 1.4,
                    "window_idx": 2,
                    "tactic": "Coordinated Attack",
                    "confidence": 0.829,
                    "entropy": 0.495,
                    "color": "#f59e0b"
                }
            ]
        }
    
    sample_path = r"D:\AdobeEditing\coordAttack.mp4"
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample video file not found on disk.")
        
    task_id = str(uuid.uuid4())[:8]
    out_filename = f"sample_analysis_{task_id}.mp4"
    
    try:
        result = video_service.process_video_file(
            input_video_path=sample_path,
            output_filename=out_filename,
            model_key=model_key,
            max_duration_sec=float(max_duration_sec)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/match-timeline")
async def get_match_timeline():
    """Returns the 5-minute Paris 2024 Olympic match timeline and predictions."""
    timeline = model_service.get_match_timeline_overview()
    return timeline


@app.get("/api/download-suite")
async def download_suite_package():
    """Generates and serves a complete VolleyTactix AI Tactical Suite ZIP bundle."""
    import zipfile
    import json

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Models and Leaderboard
        models_list = []
        for key, val in model_service.MODEL_REGISTRY.items():
            models_list.append({
                "key": key,
                "name": val["name"],
                "type": val["type"],
                "rank": val["rank"],
                "test_acc": val["test_acc"],
                "test_f1": val["test_f1"],
                "train_acc": val["train_acc"],
                "description": val["description"]
            })
        models_list.sort(key=lambda x: x["rank"])
        zf.writestr("models_benchmark_summary.json", json.dumps(models_list, indent=2))

        # Leaderboard CSV
        csv_rows = ["rank,model,family,test_accuracy,macro_f1,train_accuracy"]
        for m in models_list:
            csv_rows.append(f"{m.get('rank')},{m.get('name')},{m.get('type')},{m.get('test_acc')},{m.get('test_f1')},{m.get('train_acc')}")
        zf.writestr("lovo_leaderboard.csv", "\n".join(csv_rows))

        # 2. Timeline overview
        timeline_data = model_service.get_match_timeline_overview()
        zf.writestr("paris2024_olympic_timeline.json", json.dumps(timeline_data, indent=2))

        # 3. Sample rally file
        sample_path = os.path.join(ROOT_DIR, "training_csv_v4", "Coordinated_Attack__usa_clip_001.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as sf:
                zf.writestr("sample_rally_formation.csv", sf.read())

        # 4. Readme and instructions
        readme_text = """===============================================================
VOLLEYTACTIX AI - AUTOMATED TACTICAL INTELLIGENCE SUITE v2.4
===============================================================

Welcome to the VolleyTactix AI Tactical Coordination Analytics Suite.
This package contains:

1. models_benchmark_summary.json
   - Evaluation metrics across 13 sequence classifiers (9 ML + 4 DL)
   - Macro-F1, LOVO (Leave-One-Video-Out) Test Accuracies, and parameter counts.
   - #1 Ranked Model: HistGradientBoosting (66.8% LOVO, 0.66 Macro-F1)
   - #2 Ranked Model: InceptionTime DL (64.9% LOVO, 0.65 Macro-F1)

2. lovo_leaderboard.csv
   - Machine-readable benchmark table for publication & comparative analysis.

3. paris2024_olympic_timeline.json
   - 309 continuous tracking windows (Japan vs Argentina, Paris 2024 Olympics).
   - Timestamped tactical predictions, inter-model agreement, and entropy scores.

4. sample_rally_formation.csv
   - 29-frame normalized broadcast trajectory coordinates (X, Y) for 6 players + ball.

API & Usage:
- Web Dashboard: http://127.0.0.1:8000/
- REST API Documentation: http://127.0.0.1:8000/docs
- Live Prediction: POST /api/predict with {"model_key": "...", "sample_id": "..."}

Thank you for using VolleyTactix AI!
"""
        zf.writestr("README_TACTICAL_INTELLIGENCE.txt", readme_text)

    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=volleytactix_ai_tactical_suite.zip"}
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

