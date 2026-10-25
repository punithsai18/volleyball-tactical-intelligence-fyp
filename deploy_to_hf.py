"""
deploy_to_hf.py - One-Click Hugging Face Spaces Deployment for VolleyTactix AI.

This script helps you deploy only the required frontend and backend files
to Hugging Face Spaces without uploading heavy datasets or raw training videos.

Methods:
1. Hugging Face API: (Recommended)
   pip install huggingface_hub
   python deploy_to_hf.py --space username/volleyball-tactical-intelligence --token hf_xxx

2. Git Remote:
   git remote add space https://huggingface.co/spaces/username/volleyball-tactical-intelligence
   git push space main
"""

import os
import sys
import argparse
import subprocess
import shutil

def prepare_hf_bundle(output_dir="hf_space"):
    """Copies only the essential frontend, backend, and deployed ML model files."""
    print(f"[*] Packaging minimal deployment files into '{output_dir}'...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Copy app (frontend static + FastAPI backend)
    shutil.copytree("app", os.path.join(output_dir, "app"))

    # 2. Copy root configuration files
    for fname in ["requirements.txt", "Dockerfile", ".dockerignore", "main.py"]:
        if os.path.exists(fname):
            shutil.copy(fname, os.path.join(output_dir, fname))

    # 3. Copy fyp library (excluding tests)
    if os.path.exists("fyp"):
        shutil.copytree("fyp", os.path.join(output_dir, "fyp"), ignore=shutil.ignore_patterns("tests", "*test*", "__pycache__"))

    # 4. Copy deployed ML models
    if os.path.exists("tactical_ml_v4.joblib"):
        shutil.copy("tactical_ml_v4.joblib", os.path.join(output_dir, "tactical_ml_v4.joblib"))

    os.makedirs(os.path.join(output_dir, "ALL_13_MODELS"), exist_ok=True)
    if os.path.exists("ALL_13_MODELS/ML_models"):
        shutil.copytree("ALL_13_MODELS/ML_models", os.path.join(output_dir, "ALL_13_MODELS", "ML_models"), ignore=shutil.ignore_patterns("__pycache__"))

    # 5. Hugging Face README
    hf_readme = """---
title: VolleyTactix AI - Automated Volleyball Tactical Intelligence
emoji: 🏐
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# VolleyTactix AI — Automated Volleyball Tactical Intelligence Dashboard

Final Year Project (Group 56 • Department of Computer Science and Engineering)
Automated volleyball tactical coordination analysis from broadcast match video using computer vision, homography court normalization, 18-dim spatial invariants, and 16 learned sequence classifiers.

- **Dataset**: 948 hand-annotated rally clips across 11 broadcast matches.
- **Model Winner**: HistGradientBoosting (67.0% LOVO Test Accuracy, 0.51 Macro-F1).
- **Actions Tagged**: 8 classes (Serve, Reception, Set, Spike, Block, Dig, Pass, Free Ball).
"""
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(hf_readme)

    print(f"[+] Bundle ready in '{output_dir}'!")

def deploy_via_api(space_id: str, token: str, bundle_dir="hf_space"):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[*] Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import HfApi

    print(f"[*] Uploading '{bundle_dir}' to Hugging Face Space: {space_id}...")
    api = HfApi(token=token)
    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        folder_path=bundle_dir,
        repo_id=space_id,
        repo_type="space",
        commit_message="Deploy VolleyTactix AI frontend & backend to Hugging Face Space"
    )
    print(f"[+] Deployment completed! Your space is live at: https://huggingface.co/spaces/{space_id}")

def main():
    parser = argparse.ArgumentParser(description="Deploy VolleyTactix AI to Hugging Face Spaces")
    parser.add_argument("--space", type=str, help="Hugging Face Space ID (e.g. your-username/volleyball-tactical-ai)")
    parser.add_argument("--token", type=str, help="Hugging Face User Access Token (from https://huggingface.co/settings/tokens)")
    parser.add_argument("--bundle-only", action="store_true", help="Only prepare the hf_space folder without uploading")

    args = parser.parse_args()
    prepare_hf_bundle()

    if args.bundle_only:
        print("[+] Done. Only bundle was prepared.")
        return

    if args.space and args.token:
        deploy_via_api(args.space, args.token)
    else:
        print("\n[*] To deploy directly via API, run:")
        print("   python deploy_to_hf.py --space <your-hf-username>/<your-space-name> --token <your-hf-token>")
        print("\n[*] Alternatively, to link via Git:")
        print("   cd hf_space")
        print("   git init")
        print("   git remote add origin https://huggingface.co/spaces/<your-hf-username>/<your-space-name>")
        print("   git add .")
        print("   git commit -m 'Deploy VolleyTactix AI'")
        print("   git push -u origin main --force")

if __name__ == "__main__":
    main()
