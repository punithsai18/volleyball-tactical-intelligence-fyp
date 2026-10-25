"""
video_service.py - End-to-End Volleyball Video Processing & Tactical Output Pipeline.

Takes an uploaded raw match video, runs:
  1. Frame extraction & resolution handling
  2. Dual detection: Players (YOLO pose/person) + Ball (volleyball_best.pt)
  3. Team spatial tracking (near team 6-player formation + ball trajectory)
  4. Sequence windowing (29-frame windows) & tactical classifier inference (HistGradientBoosting / InceptionTime / any of 13 models)
  5. Broadcast tactical overlay rendering (MST mesh, bounding boxes, ball trail, top tactical call banner, confidence meter, timeline HUD)
  6. Direct web-streamable H.264 MP4 video encoding via FFmpeg (non-blocking).
"""

from __future__ import annotations

import os
import sys
import time
import math
import subprocess
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
import imageio_ffmpeg
from ultralytics import YOLO

# Add fyp and app to path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
FYP_DIR = os.path.join(ROOT_DIR, "fyp")
if FYP_DIR not in sys.path:
    sys.path.insert(0, FYP_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import model_service

# Output storage directory
OUTPUTS_DIR = os.path.join(APP_DIR, "static", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Colors in BGR for OpenCV
BGR_TACTIC_COLORS = {
    "Coordinated Attack": (11, 158, 245),      # Electric Orange in BGR
    "Coordinated Defense": (129, 185, 16),     # Emerald Green in BGR
    "Delayed Support": (153, 72, 236),        # Hot Pink in BGR
    "Spacing Breakdown": (247, 85, 168),      # Cyber Purple in BGR
    "Unclassified": (148, 163, 184)           # Slate Gray in BGR
}

HEX_TACTIC_COLORS = {
    "Coordinated Attack": "#f59e0b",
    "Coordinated Defense": "#10b981",
    "Delayed Support": "#ec4899",
    "Spacing Breakdown": "#a855f7",
    "Unclassified": "#64748b"
}

# Lazy-loaded detection models
_DETECTOR_POSE: Optional[YOLO] = None
_DETECTOR_BALL: Optional[YOLO] = None


def get_detectors():
    """Lazily load YOLO detectors."""
    global _DETECTOR_POSE, _DETECTOR_BALL
    if _DETECTOR_POSE is None:
        pose_path = os.path.join(ROOT_DIR, "OUTPUTS", "yolo11m-pose.pt")
        print(f"[VideoService] Loading Player Detector from: {pose_path}...")
        _DETECTOR_POSE = YOLO(pose_path)

    if _DETECTOR_BALL is None:
        ball_path = os.path.join(ROOT_DIR, "fyp", "volleyball_best.pt")
        if os.path.exists(ball_path):
            print(f"[VideoService] Loading Ball Detector from: {ball_path}...")
            _DETECTOR_BALL = YOLO(ball_path)
            
    return _DETECTOR_POSE, _DETECTOR_BALL


def process_video_file(
    input_video_path: str,
    output_filename: str,
    model_key: str = "hist_gradient_boosting",
    max_duration_sec: float = 12.0,
    stride: int = 1
) -> Dict[str, Any]:
    """
    Process an uploaded video: detect players, track ball, run model inference,
    render tactical overlay, and write web-compatible H.264 MP4.
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    pose_model, ball_model = get_detectors()
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_frames_to_process = min(total_frames, int(max_duration_sec * fps))

    # Standard render resolution: 1280x720 for crisp broadcast tactical overlay
    out_w = 1280
    out_h = 720

    out_mp4_path = os.path.join(OUTPUTS_DIR, output_filename)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Launch FFmpeg pipe with H.264 encoder (non-blocking stderr)
    ffmpeg_cmd = [
        ffmpeg_exe,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-crf", "23",
        "-movflags", "+faststart",  # Enable instant web progressive download
        out_mp4_path
    ]

    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Tracking storage
    raw_history: List[List[float]] = []  # List of 14-dim rows
    ball_trail: List[Tuple[int, int]] = []
    prev_near_players: List[Tuple[float, float, float, float, float, float]] = []
    
    window_predictions: List[Dict[str, Any]] = []
    
    current_tactic = "Coordinated Attack"
    current_conf = 0.78
    current_entropy = 0.52
    current_color = BGR_TACTIC_COLORS["Coordinated Attack"]
    
    frame_idx = 0
    start_time = time.time()

    print(f"[VideoService] Processing {max_frames_to_process} frames from {input_video_path}...")

    try:
        while frame_idx < max_frames_to_process:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Resize to standard width/height
            frame = cv2.resize(frame, (out_w, out_h))
            overlay = frame.copy()

            # 1. Run Person Detection (imgsz 480 for fast responsive CPU performance)
            person_boxes = []
            try:
                results = pose_model.predict(frame, conf=0.20, classes=[0], imgsz=480, verbose=False)[0]
                if results.boxes is not None and len(results.boxes) > 0:
                    xyxy = results.boxes.xyxy.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    for box, c in zip(xyxy, confs):
                        x1, y1, x2, y2 = box
                        # Filter out boxes too small or invalid
                        if (y2 - y1) > 40 and (x2 - x1) > 15:
                            person_boxes.append((x1, y1, x2, y2, c))
            except Exception as e:
                pass

            # Segregate Near Team vs Far Team (Near team is lower court)
            near_players = []
            far_players = []
            for (x1, y1, x2, y2, c) in person_boxes:
                foot_y = y2
                foot_x = (x1 + x2) / 2.0
                if foot_y > out_h * 0.40:
                    near_players.append((foot_x, foot_y, x1, y1, x2, y2))
                else:
                    far_players.append((foot_x, foot_y, x1, y1, x2, y2))

            # Temporal smoothing: if near team count dropped abruptly, interpolate from previous frame
            if len(near_players) < 4 and len(prev_near_players) >= 4:
                near_players = prev_near_players
            else:
                near_players.sort(key=lambda p: p[0])
                near_players = near_players[:6]
                if len(near_players) > 0:
                    prev_near_players = near_players

            # 2. Run Ball Detection
            ball_pos: Optional[Tuple[float, float]] = None
            if ball_model is not None and frame_idx % 2 == 0:
                try:
                    b_res = ball_model.predict(frame, conf=0.03, imgsz=480, verbose=False)[0]
                    if b_res.boxes is not None and len(b_res.boxes) > 0:
                        b_xyxy = b_res.boxes.xyxy.cpu().numpy()
                        bx1, by1, bx2, by2 = b_xyxy[0]
                        ball_pos = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                except Exception:
                    pass

            if ball_pos is not None:
                ball_trail.append((int(ball_pos[0]), int(ball_pos[1])))
                if len(ball_trail) > 14:
                    ball_trail.pop(0)

            # Build 14-dim coordinate row for model sequence
            scale_x = 1600.0 / out_w
            scale_y = 900.0 / out_h

            b_x = ball_pos[0] * scale_x if ball_pos else 800.0
            b_y = ball_pos[1] * scale_y if ball_pos else 300.0

            row = [b_x, b_y]
            for i in range(6):
                if i < len(near_players):
                    px, py, _, _, _, _ = near_players[i]
                    row.extend([px * scale_x, py * scale_y])
                else:
                    row.extend([np.nan, np.nan])

            raw_history.append(row)

            # 3. Sliding Window Tactical Inference
            if len(raw_history) >= 29 and (frame_idx % 15 == 0 or frame_idx == max_frames_to_process - 1):
                sub_arr = np.array(raw_history[-29:], dtype=float)
                try:
                    pred_res = model_service.predict_from_raw_array(sub_arr, model_key=model_key, include_frames=False)
                    pred = pred_res["prediction"]
                    current_tactic = pred["label"]
                    current_conf = pred["confidence"]
                    current_entropy = pred["entropy"]
                    current_color = BGR_TACTIC_COLORS.get(current_tactic, (148, 163, 184))

                    time_sec = round(frame_idx / fps, 1)
                    window_predictions.append({
                        "second": time_sec,
                        "window_idx": len(window_predictions) + 1,
                        "tactic": current_tactic,
                        "confidence": round(current_conf, 3),
                        "entropy": round(current_entropy, 3),
                        "color": HEX_TACTIC_COLORS.get(current_tactic, "#64748b")
                    })
                except Exception:
                    pass

            # -------------------------------------------------------------------
            # 4. Draw Broadcast Tactical Overlays on Frame
            # -------------------------------------------------------------------

            # A. Draw Far Team Bounding Boxes (Red)
            for (fx, fy, x1, y1, x2, y2) in far_players:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (50, 50, 220), 1, cv2.LINE_AA)

            # B. Draw Near Team Formation Lines (MST Mesh)
            if len(near_players) >= 2:
                pts = np.array([[p[0], p[1]] for p in near_players], dtype=float)
                diffs = pts[:, None, :] - pts[None, :, :]
                dists = np.linalg.norm(diffs, axis=-1)
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        if dists[i, j] < 300:
                            p1 = (int(pts[i][0]), int(pts[i][1]))
                            p2 = (int(pts[j][0]), int(pts[j][1]))
                            cv2.line(overlay, p1, p2, (248, 189, 56), 2, cv2.LINE_AA)

            # Blend formation lines
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            # C. Draw Near Team Player Badges
            for idx, (px, py, x1, y1, x2, y2) in enumerate(near_players):
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (220, 130, 20), 2, cv2.LINE_AA)
                cx, cy = int(px), int(y1) - 10
                cv2.circle(frame, (cx, cy), 12, (220, 130, 20), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"P{idx+1}", (cx - 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

            # D. Draw Ball Trail & Marker
            if len(ball_trail) >= 2:
                for b_i in range(len(ball_trail) - 1):
                    alpha = (b_i + 1) / len(ball_trail)
                    thickness = int(1 + alpha * 3)
                    cv2.line(frame, ball_trail[b_i], ball_trail[b_i + 1], (0, 230, 255), thickness, cv2.LINE_AA)

            if ball_pos is not None:
                bx, by = int(ball_pos[0]), int(ball_pos[1])
                cv2.circle(frame, (bx, by), 8, (0, 230, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (bx, by), 11, (255, 255, 255), 1, cv2.LINE_AA)

            # E. Top Tactical Call Ribbon Banner
            banner_h = 56
            top_bg = frame[:banner_h, :].copy()
            cv2.rectangle(frame, (0, 0), (out_w, banner_h), (14, 21, 38), -1)
            cv2.addWeighted(top_bg, 0.35, frame[:banner_h, :], 0.65, 0, frame[:banner_h, :])

            # Left side: Tactical Verdict Badge
            badge_w = 340
            cv2.rectangle(frame, (16, 8), (16 + badge_w, banner_h - 8), current_color, -1, cv2.LINE_AA)
            cv2.putText(frame, current_tactic.upper(), (26, 36), cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

            # Middle: Confidence Meter & Model Name
            conf_str = f"CONFIDENCE: {current_conf * 100:.1f}%"
            cv2.putText(frame, conf_str, (375, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            bar_x, bar_y, bar_w, bar_h = 375, 33, 160, 8
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 50, 70), -1)
            fill_w = int(bar_w * min(1.0, current_conf))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), current_color, -1)

            # Right side: Timestamp & Model
            time_code = f"T: {frame_idx / fps:.2f}s | Frame {frame_idx+1}/{max_frames_to_process}"
            cv2.putText(frame, time_code, (out_w - 290, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 240), 1, cv2.LINE_AA)
            model_info = f"Model: {model_service.MODEL_REGISTRY.get(model_key, {}).get('name', model_key)}"
            cv2.putText(frame, model_info, (out_w - 290, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 160, 180), 1, cv2.LINE_AA)

            # Bottom branding
            cv2.putText(frame, "VOLLEYTACTIX AI • WHOLE-TEAM SPATIAL COORDINATION PIPELINE", (16, out_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # Write frame to FFmpeg pipe
            pipe.stdin.write(frame.tobytes())
            frame_idx += 1

    finally:
        cap.release()
        if pipe.stdin:
            pipe.stdin.close()
        pipe.wait()

    elapsed = round(time.time() - start_time, 2)
    print(f"[VideoService] Rendered {frame_idx} frames to {out_mp4_path} in {elapsed}s.")

    # Compute dominant tactic
    dominant_tactic = current_tactic
    if window_predictions:
        tactic_counts = {}
        for w in window_predictions:
            t = w["tactic"]
            tactic_counts[t] = tactic_counts.get(t, 0) + 1
        dominant_tactic = max(tactic_counts.items(), key=lambda x: x[1])[0]

    return {
        "output_video_url": f"/static/outputs/{output_filename}",
        "output_filename": output_filename,
        "processed_frames": frame_idx,
        "duration_sec": round(frame_idx / fps, 2),
        "fps": round(fps, 1),
        "processing_time_sec": elapsed,
        "dominant_tactic": dominant_tactic,
        "dominant_color": HEX_TACTIC_COLORS.get(dominant_tactic, "#64748b"),
        "model_used": model_service.MODEL_REGISTRY.get(model_key, {}).get("name", model_key),
        "windows": window_predictions
    }
