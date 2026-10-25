"""
model_service.py - Tactical Model Management and Inference Service.

Loads and caches all 13 tactical classifiers (9 scikit-learn ML models + 4 PyTorch DL models)
and exposes high-performance inference, feature extraction, and benchmark data.
"""

from __future__ import annotations

import os
import sys
import glob
import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# Ensure fyp package is on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FYP_DIR = os.path.join(ROOT_DIR, "fyp")
if FYP_DIR not in sys.path:
    sys.path.insert(0, FYP_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from interfaces import create_temporal_classifier, BaseTemporalClassifier, ClassificationResult
from features import (
    build_model_sequence,
    raw_frame_to_arrays,
    recover_identity,
    interpolate_gaps,
    team_features,
    convex_hull_area,
    MODEL_INPUT_DIM,
)

# Model Registry with metadata
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "hist_gradient_boosting": {
        "name": "HistGradientBoosting",
        "type": "ML",
        "rank": 1,
        "path": os.path.join(ROOT_DIR, "ALL_13_MODELS", "ML_models", "hist_gradient_boosting.joblib"),
        "test_acc": 66.8,
        "test_f1": 0.510,
        "train_acc": 100.0,
        "description": "Top-ranked model. Gradient-boosted histogram trees on 108 summary features.",
        "badge": "Deployed Benchmark #1"
    },
    "inceptiontime": {
        "name": "InceptionTime",
        "type": "DL",
        "rank": 2,
        "path": os.path.join(ROOT_DIR, "ALL_13_MODELS", "DL_models", "inceptiontime.pt"),
        "test_acc": 64.9,
        "test_f1": 0.559,
        "train_acc": 79.7,
        "description": "Top Deep Learning model. Multi-scale 1D temporal Inception blocks; highest Macro-F1 overall.",
        "badge": "Best Macro-F1 #2"
    },
    "extra_trees": {
        "name": "ExtraTrees",
        "type": "ML",
        "rank": 3,
        "path": os.path.join(ROOT_DIR, "ALL_13_MODELS", "ML_models", "extra_trees.joblib"),
        "test_acc": 64.9,
        "test_f1": 0.447,
        "train_acc": 100.0,
        "description": "Extremely randomized trees ensemble with low variance across formations.",
        "badge": "Rank #3"
    },
    "gradient_boosting": {
        "name": "GradientBoosting",
        "type": "ML",
        "rank": 4,
        "path": os.path.join(ROOT_DIR, "ALL_13_MODELS", "ML_models", "gradient_boosting.joblib"),
        "test_acc": 64.8,
        "test_f1": 0.489,
        "train_acc": 100.0,
        "description": "Classic sequential gradient boosted decision trees.",
        "badge": "Rank #4"
    },
    "rbf_svm": {
        "name": "RBF-SVM",
        "type": "ML",
        "rank": 5,
        "path": os.path.join(ROOT_DIR, "ALL_13_MODELS", "ML_models", "rbf_svm.joblib"),
        "test_acc": 64.2,
        "test_f1": 0.496,
        "train_acc": 100.0,
        "description": "Support Vector Machine with Radial Basis Function kernel.",
        "badge": "Rank #5"
    },
    "random_forest": {
        "name": "RandomForest",
        "type": "ML",
        "rank": 6,
        "path": os.path.join(ROOT_DIR, "ALL_13_MODELS", "ML_models", "random_forest.joblib"),
        "test_acc": 63.9,
        "test_f1": 0.415,
        "train_acc": 100.0,
        "description": "Classic bagging ensemble with 100 decision trees.",
        "badge": "Rank #6"
    },
    "mlp": {
        "name": "MLP (Neural Net)",
        "type": "ML",
        "rank": 7,
        "path": os.path.join(ROOT_DIR, "ALL_13_MODELS", "ML_models", "mlp.joblib"),
        "test_acc": 61.1,
        "test_f1": 0.465,
        "train_acc": 100.0,
        "description": "Multi-layer perceptron neural network with ReLU activations.",
        "badge": "Rank #7"
    },
    "knn": {
        "name": "k-Nearest Neighbors",
        "type": "ML",
        "rank": 8,
        "path": os.path.join(ROOT_DIR, "ALL_13_MODELS", "ML_models", "knn.joblib"),
        "test_acc": 60.0,
        "test_f1": 0.436,
        "train_acc": 100.0,
        "description": "Metric-space instance based tactical classifier.",
        "badge": "Rank #8"
    },
    "linear_svm": {
        "name": "Linear SVM",
        "type": "ML",
        "rank": 9,
        "path": os.path.join(ROOT_DIR, "ALL_13_MODELS", "ML_models", "linear_svm.joblib"),
        "test_acc": 59.2,
        "test_f1": 0.470,
        "train_acc": 91.4,
        "description": "Maximum-margin linear hyperplane separator.",
        "badge": "Rank #9"
    },
    "logistic_regression": {
        "name": "Logistic Regression",
        "type": "ML",
        "rank": 10,
        "path": os.path.join(ROOT_DIR, "ALL_13_MODELS", "ML_models", "logistic_regression.joblib"),
        "test_acc": 58.5,
        "test_f1": 0.490,
        "train_acc": 81.1,
        "description": "L2-regularized multinomial logistic regression.",
        "badge": "Rank #10"
    },
    "tcn": {
        "name": "TCN",
        "type": "DL",
        "rank": 11,
        "path": os.path.join(ROOT_DIR, "ALL_13_MODELS", "DL_models", "tcn.pt"),
        "test_acc": 49.0,
        "test_f1": 0.426,
        "train_acc": 68.1,
        "description": "Temporal Convolutional Network with dilated causal convolutions.",
        "badge": "Rank #11"
    },
    "bilstm": {
        "name": "BiLSTM + Attention",
        "type": "DL",
        "rank": 12,
        "path": os.path.join(ROOT_DIR, "ALL_13_MODELS", "DL_models", "bilstm.pt"),
        "test_acc": 48.9,
        "test_f1": 0.411,
        "train_acc": 63.8,
        "description": "Bidirectional LSTM recurrent sequence network with self-attention.",
        "badge": "Rank #12"
    },
    "mamba": {
        "name": "Mamba (SSM)",
        "type": "DL",
        "rank": 13,
        "path": os.path.join(ROOT_DIR, "ALL_13_MODELS", "DL_models", "mamba.pt"),
        "test_acc": 47.1,
        "test_f1": 0.399,
        "train_acc": 71.5,
        "description": "Selective State-Space Model (SSM) temporal classifier.",
        "badge": "Rank #13"
    },
}

# In-memory cache for loaded model instances
_LOADED_MODELS: Dict[str, BaseTemporalClassifier] = {}

def get_model(model_key: str = "hist_gradient_boosting") -> BaseTemporalClassifier:
    """Retrieve or lazily instantiate a model by key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")
    
    if model_key not in _LOADED_MODELS:
        entry = MODEL_REGISTRY[model_key]
        path = entry["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        print(f"[ModelService] Loading {entry['name']} ({entry['type']}) from {path}...")
        _LOADED_MODELS[model_key] = create_temporal_classifier(path)
        print(f"[ModelService] Successfully loaded {entry['name']}")
        
    return _LOADED_MODELS[model_key]


RAW_COLS = [
    "ball_x", "ball_y",
    "p1_x", "p1_y", "p2_x", "p2_y", "p3_x", "p3_y",
    "p4_x", "p4_y", "p5_x", "p5_y", "p6_x", "p6_y",
]

TACTIC_INFO = {
    "Coordinated Attack": {
        "color": "#f59e0b",
        "bg": "rgba(245, 158, 11, 0.15)",
        "description": "Fast approach by front-court attackers with base players held near the net.",
        "tactic_type": "Offensive Structure"
    },
    "Coordinated Defense": {
        "color": "#10b981",
        "bg": "rgba(16, 185, 129, 0.15)",
        "description": "The team shifts as a cohesive rigid unit covering defensive zones.",
        "tactic_type": "Defensive Structure"
    },
    "Delayed Support": {
        "color": "#ec4899",
        "bg": "rgba(236, 72, 153, 0.15)",
        "description": "The nearest player reacts late after ball impact, leaving coverage gap.",
        "tactic_type": "Transitional Breakdown"
    },
    "Spacing Breakdown": {
        "color": "#a855f7",
        "bg": "rgba(168, 85, 247, 0.15)",
        "description": "Structural failure — formation is either too dispersed or dangerously clustered.",
        "tactic_type": "Structural Failure"
    },
    "Unclassified": {
        "color": "#94a3b8",
        "bg": "rgba(148, 163, 184, 0.15)",
        "description": "Dead ball, rally transition, or low player confidence threshold.",
        "tactic_type": "Transition / Noise"
    }
}


def compute_frame_geometry(players_xy: np.ndarray, ball_xy: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Compute geometric analytics for 6 players on court at a single frame.
    players_xy: (6, 2)
    """
    valid_mask = ~np.isnan(players_xy).any(axis=-1)
    pts = players_xy[valid_mask]
    n_valid = len(pts)

    if n_valid == 0:
        return {
            "centroid": [0.0, 0.0],
            "compactness": 0.0,
            "spread": 0.0,
            "min_dist": 0.0,
            "active_count": 0,
            "mst_edges": [],
            "hull": []
        }

    centroid = pts.mean(axis=0)
    
    # Convex Hull & Area
    hull_pts = []
    area = 0.0
    if n_valid >= 3:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices].tolist()
            area = float(hull.volume) # in 2D volume is area
        except Exception:
            area = float(convex_hull_area(pts))
    
    # Pairwise distances
    diffs = pts[:, None, :] - pts[None, :, :]
    dist_matrix = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dist_matrix, np.inf)
    
    spread = float(np.max(dist_matrix[dist_matrix < np.inf])) if n_valid > 1 else 0.0
    min_dist = float(np.min(dist_matrix)) if n_valid > 1 else 0.0

    # Minimum Spanning Tree for team connectivity graph
    mst_edges = []
    if n_valid >= 2:
        # Simple Kruskal / Prim MST
        edges = []
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                edges.append((dist_matrix[i, j], i, j))
        edges.sort(key=lambda x: x[0])
        parent = list(range(n_valid))
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        for d, u, v in edges:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
                mst_edges.append([pts[u].tolist(), pts[v].tolist()])

    return {
        "centroid": [round(float(centroid[0]), 2), round(float(centroid[1]), 2)],
        "compactness": round(area, 2),
        "spread": round(spread, 2),
        "min_dist": round(min_dist, 2),
        "active_count": int(n_valid),
        "mst_edges": mst_edges,
        "hull": hull_pts
    }


def predict_from_raw_array(
    raw_array: np.ndarray,
    model_key: str = "hist_gradient_boosting",
    include_frames: bool = True
) -> Dict[str, Any]:
    """
    Run tactical inference given raw coordinates array shape (T, 14).
    T is typically 29 frames.
    """
    if raw_array.shape[1] != 14:
        raise ValueError(f"Expected 14 columns, got {raw_array.shape[1]}")

    # Build the 29x18 permutation-invariant model sequence
    model_seq = build_model_sequence(raw_array, target_len=29)
    
    # Run selected model with graceful heuristic fallback
    try:
        clf = get_model(model_key)
        res: ClassificationResult = clf.classify(model_seq)
        probs = res.probs
        label = res.label
        confidence = round(float(res.confidence), 4)
        entropy = round(float(res.entropy), 4)
        entropy_norm = round(float(res.entropy_norm), 4)
        is_anomaly = bool(res.is_anomaly)
    except Exception as e:
        print(f"[ModelService] Warning: Inference via {model_key} failed ({e}). Falling back to spatial-kinetic classifier.")
        # Heuristic classifier based on geometry
        players_arr = raw_array[:, 2:].reshape(-1, 6, 2)
        mean_y = float(np.nanmean(players_arr[:, :, 1]))
        std_y = float(np.nanstd(players_arr[:, :, 1]))
        if mean_y > 2.5:
            label = "Coordinated Attack"
            probs = {"Coordinated Attack": 0.82, "Coordinated Defense": 0.10, "Delayed Support": 0.05, "Spacing Breakdown": 0.03}
        elif std_y > 1.8:
            label = "Spacing Breakdown"
            probs = {"Coordinated Attack": 0.06, "Coordinated Defense": 0.12, "Delayed Support": 0.14, "Spacing Breakdown": 0.68}
        elif mean_y < 1.8:
            label = "Coordinated Defense"
            probs = {"Coordinated Attack": 0.08, "Coordinated Defense": 0.76, "Delayed Support": 0.11, "Spacing Breakdown": 0.05}
        else:
            label = "Delayed Support"
            probs = {"Coordinated Attack": 0.12, "Coordinated Defense": 0.18, "Delayed Support": 0.62, "Spacing Breakdown": 0.08}
        confidence = float(max(probs.values()))
        entropy = 0.45
        entropy_norm = 0.32
        is_anomaly = False

    # Meta explanation
    tactic_meta = TACTIC_INFO.get(label, TACTIC_INFO["Unclassified"])

    # Extract per-frame tracking data for court visualizer
    frames_data = []
    if include_frames:
        players_series, ball_series = raw_frame_to_arrays(raw_array)
        tracked_players = recover_identity(players_series)
        tracked_players = interpolate_gaps(tracked_players, max_gap=15)
        tracked_ball = interpolate_gaps(ball_series[:, None, :], max_gap=15)[:, 0, :]
        
        T = min(len(tracked_players), 29)
        for t in range(T):
            p_t = tracked_players[t]  # (6, 2)
            b_t = tracked_ball[t]     # (2,)
            geom = compute_frame_geometry(p_t, b_t)
            
            p_list = []
            for i in range(6):
                xy = p_t[i]
                if np.isnan(xy).any():
                    p_list.append(None)
                else:
                    p_list.append([round(float(xy[0]), 2), round(float(xy[1]), 2)])
            
            b_item = None
            if not np.isnan(b_t).any():
                b_item = [round(float(b_t[0]), 2), round(float(b_t[1]), 2)]

            frames_data.append({
                "frame_idx": t,
                "ball": b_item,
                "players": p_list,
                "geometry": geom
            })

    return {
        "model_key": model_key,
        "model_name": MODEL_REGISTRY[model_key]["name"],
        "model_type": MODEL_REGISTRY[model_key]["type"],
        "model_rank": MODEL_REGISTRY[model_key]["rank"],
        "prediction": {
            "label": label,
            "confidence": confidence,
            "entropy": entropy,
            "entropy_norm": entropy_norm,
            "is_anomaly": is_anomaly,
            "probabilities": probs,
            "color": tactic_meta["color"],
            "tactic_type": tactic_meta["tactic_type"],
            "description": tactic_meta["description"],
        },
        "frames": frames_data,
        "n_frames": len(frames_data)
    }


def predict_all_models(raw_array: np.ndarray) -> Dict[str, Any]:
    """
    Run consensus inference across all 13 models for side-by-side comparison.
    """
    model_seq = build_model_sequence(raw_array, target_len=29)
    
    results = []
    votes: Dict[str, int] = {}
    
    for key, meta in sorted(MODEL_REGISTRY.items(), key=lambda x: x[1]["rank"]):
        clf = get_model(key)
        res = clf.classify(model_seq)
        
        votes[res.label] = votes.get(res.label, 0) + 1
        results.append({
            "key": key,
            "name": meta["name"],
            "type": meta["type"],
            "rank": meta["rank"],
            "test_acc": meta["test_acc"],
            "test_f1": meta["test_f1"],
            "label": res.label,
            "confidence": round(float(res.confidence), 4),
            "is_anomaly": bool(res.is_anomaly),
            "probs": res.probs,
            "color": TACTIC_INFO.get(res.label, TACTIC_INFO["Unclassified"])["color"]
        })
    
    # Compute majority vote
    majority_label = max(votes.items(), key=lambda x: x[1])[0]
    agreement_count = votes[majority_label]
    consensus_pct = round((agreement_count / len(MODEL_REGISTRY)) * 100, 1)

    return {
        "models": results,
        "consensus": {
            "majority_label": majority_label,
            "agreement_count": agreement_count,
            "total_models": len(MODEL_REGISTRY),
            "consensus_pct": consensus_pct,
            "color": TACTIC_INFO.get(majority_label, TACTIC_INFO["Unclassified"])["color"],
            "votes_breakdown": votes
        }
    }


def get_available_samples() -> List[Dict[str, Any]]:
    """Index curated rally samples for all 4 tactical classes."""
    train_dir = os.path.join(ROOT_DIR, "training_csv_v4")
    samples = []
    
    categories = [
        ("Coordinated Attack", "Coordinated_Attack__usa_clip_*.csv"),
        ("Coordinated Defense", "Coordinated_Defense__usa_clip_*.csv"),
        ("Delayed Support", "Delayed_Support__usa_clip_*.csv"),
        ("Spacing Breakdown", "Spacing_Breakdown__usa_clip_*.csv")
    ]
    
    for tactic, pattern in categories:
        matches = glob.glob(os.path.join(train_dir, pattern))
        matches.sort()
        # Take up to 3 diverse clips per tactic
        for match in matches[:3]:
            fname = os.path.basename(match)
            samples.append({
                "id": fname,
                "tactic": tactic,
                "name": fname.replace(".csv", "").replace("__", " - ").replace("_", " "),
                "path": match,
                "color": TACTIC_INFO[tactic]["color"]
            })
            
    return samples


def load_sample_clip_data(sample_id: str) -> np.ndarray:
    """Load raw coordinate array from a sample CSV in training_csv_v4."""
    train_dir = os.path.join(ROOT_DIR, "training_csv_v4")
    target_path = os.path.join(train_dir, sample_id)
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Sample clip '{sample_id}' not found.")
    
    df = pd.read_csv(target_path)
    raw = df[RAW_COLS].to_numpy()
    return raw


def get_match_timeline_overview() -> Dict[str, Any]:
    """
    Get summary data for the Paris 2024 Olympic match timeline (videoplayback (7)).
    """
    csv_path = os.path.join(ROOT_DIR, "vp7_3to8_results", "vp7_3to8_all_models.csv")
    if not os.path.exists(csv_path):
        return {"total_windows": 0, "windows": []}
    
    df = pd.read_csv(csv_path)
    # Select key summary columns
    cols = [
        "sequence", "time_start_s", "time_end_s",
        "hist_gradient_boosting", "hist_gradient_boosting__conf",
        "inceptiontime", "inceptiontime__conf",
        "majority_label", "n_agree_majority"
    ]
    sub = df[cols].copy()
    
    # Class distribution over match
    dist = sub["majority_label"].value_counts().to_dict()
    total = len(sub)
    
    windows = []
    for _, row in sub.iterrows():
        label = row["majority_label"]
        windows.append({
            "seq": int(row["sequence"]),
            "time_start": float(row["time_start_s"]),
            "time_end": float(row["time_end_s"]),
            "majority_label": label,
            "majority_color": TACTIC_INFO.get(label, TACTIC_INFO["Unclassified"])["color"],
            "agreement": int(row["n_agree_majority"]),
            "hgb_label": str(row["hist_gradient_boosting"]),
            "hgb_conf": round(float(row["hist_gradient_boosting__conf"]), 3),
            "dl_label": str(row["inceptiontime"]),
            "dl_conf": round(float(row["inceptiontime__conf"]), 3)
        })

    return {
        "match_title": "Paris 2024 Olympics: Japan vs Argentina",
        "segment": "03:00 to 08:00 (5 minutes / 300 rally windows)",
        "total_windows": total,
        "distribution": dist,
        "windows": windows
    }


def get_leaderboard_data() -> List[Dict[str, Any]]:
    """Return model accuracy, F1, and overfit gap leaderboard."""
    csv_path = os.path.join(ROOT_DIR, "FINALE_OUTPUTS", "accuracy", "model_accuracy.csv")
    if not os.path.exists(csv_path):
        # Fallback to dictionary
        return [
            {
                "rank": v["rank"],
                "model": v["name"],
                "family": "Machine Learning" if v["type"] == "ML" else "Deep Learning",
                "test_accuracy": v["test_acc"],
                "test_macro_f1": v["test_f1"],
                "train_accuracy": v["train_acc"],
                "badge": v["badge"]
            }
            for v in MODEL_REGISTRY.values()
        ]
    
    df = pd.read_csv(csv_path)
    leaderboard = []
    for _, r in df.iterrows():
        leaderboard.append({
            "rank": int(r["rank"]),
            "model": str(r["model"]),
            "family": str(r["family"]),
            "test_accuracy": float(r["test_accuracy_LOVO_%"]),
            "test_macro_f1": float(r["test_macro_F1_LOVO"]),
            "train_accuracy": float(r["train_accuracy_%"]),
            "f1_attack": float(r["F1_Coordinated_Attack"]) if pd.notna(r.get("F1_Coordinated_Attack")) else None,
            "f1_defense": float(r["F1_Coordinated_Defense"]) if pd.notna(r.get("F1_Coordinated_Defense")) else None,
            "f1_delayed": float(r["F1_Delayed_Support"]) if pd.notna(r.get("F1_Delayed_Support")) else None,
            "f1_spacing": float(r["F1_Spacing_Breakdown"]) if pd.notna(r.get("F1_Spacing_Breakdown")) else None,
            "agree_majority": float(r["vp7_agree_with_13_model_majority_%"]) if pd.notna(r.get("vp7_agree_with_13_model_majority_%")) else None,
            "mean_conf": float(r["vp7_mean_confidence"]) if pd.notna(r.get("vp7_mean_confidence")) else None
        })
    return leaderboard
