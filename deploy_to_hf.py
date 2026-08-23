"""
Pusher-v5 PPO // Hugging Face Hub & Spaces One-Click Deployer
--------------------------------------------------------------
Automates uploading the trained Pusher-v5 PPO model, interactive web telemetry cockpit,
video checkpoints, and evaluation artifacts to Hugging Face Model Hub / Spaces.

Usage:
    # 1. Standard deploy (Uploads all project files to Hugging Face)
    python deploy_to_hf.py

    # 2. Deploy to a custom repository name
    python deploy_to_hf.py --repo-name my-pusher-ppo

    # 3. Create private repository
    python deploy_to_hf.py --private

    # 4. Create repository only without uploading files
    python deploy_to_hf.py --create-only
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, get_token, login

DEFAULT_REPO_NAME = "pusher-v5-ppo"


def parse_args():
    parser = argparse.ArgumentParser(description="One-Click Deployer for Pusher-v5 PPO Hub")
    parser.add_argument("--repo-name", type=str, default=DEFAULT_REPO_NAME,
                        help=f"Hugging Face repository name (default: {DEFAULT_REPO_NAME})")
    parser.add_argument("--repo-type", type=str, default="model", choices=["model", "space"],
                        help="Repository type: 'model' (Model Hub) or 'space' (Interactive Space)")
    parser.add_argument("--space-sdk", type=str, default="docker", choices=["docker", "static", "gradio"],
                        help="Space SDK if repo-type is 'space' (default: docker)")
    parser.add_argument("--private", action="store_true",
                        help="Create repository as private (default: Public)")
    parser.add_argument("--create-only", action="store_true",
                        help="Only create the repository on Hugging Face without uploading files")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face User Access Token (with WRITE permission)")
    return parser.parse_args()


def check_auth(token: str = None) -> str:
    """Verifies Hugging Face authentication token."""
    if token:
        login(token=token)
        return token
    
    existing_token = get_token()
    if existing_token:
        return existing_token
    
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        login(token=env_token)
        return env_token
    
    print("\n" + "=" * 65)
    print(" [!] Hugging Face Authentication Required")
    print("=" * 65)
    print(" Please provide your Hugging Face Access Token with WRITE permission.")
    print(" You can get your token from: https://huggingface.co/settings/tokens")
    print("=" * 65 + "\n")
    
    token_input = input(" Enter your Hugging Face Token: ").strip()
    if not token_input:
        print("[ERROR] Token cannot be empty. Deployment aborted.")
        sys.exit(1)
    
    login(token=token_input)
    return token_input


def main():
    args = parse_args()
    token = check_auth(args.token)
    api = HfApi(token=token)

    try:
        user_info = api.whoami()
        username = user_info["name"]
    except Exception as e:
        print(f"[ERROR] Failed to fetch user info with provided token: {e}")
        sys.exit(1)

    repo_id = f"{username}/{args.repo_name}"
    print("\n" + "=" * 65)
    print(" 🦾 Pusher-v5 PPO // Hugging Face Deployment Pipeline")
    print("=" * 65)
    print(f" • Target User : {username}")
    print(f" • Repo ID     : {repo_id}")
    print(f" • Repo Type   : {args.repo_type}")
    print(f" • Visibility  : {'Private' if args.private else 'Public'}")
    print("=" * 65)

    # 1. Create or connect to repository
    try:
        print(f"\n[1/2] Creating/Connecting repository on Hugging Face: {repo_id} ...")
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type=args.repo_type,
            private=args.private,
            space_sdk=args.space_sdk if args.repo_type == "space" else None,
            exist_ok=True
        )
        print(f"  --> Repository ready: {repo_url}")
    except Exception as e:
        print(f"[ERROR] Repository creation failed: {e}")
        sys.exit(1)

    if args.create_only:
        print("\n[COMPLETE] '--create-only' mode selected. Repository created successfully.")
        return

    # 2. Upload project folder
    root_dir = Path(__file__).resolve().parent
    print(f"\n[2/2] Uploading project files from {root_dir} to {repo_id} ...")

    ignore_patterns = [
        "__pycache__/*",
        "*.pyc",
        ".git/*",
        ".gitignore",
        ".venv/*",
        "venv/*",
        "env/*",
        "*.log",
        ".system_generated/*",
        ".tempmediaStorage/*",
        ".cursor/*",
        ".cursorrules*",
        "DOCS_*",
        "eval_results/*",
        ".vscode/*"
    ]

    try:
        api.upload_folder(
            folder_path=str(root_dir),
            repo_id=repo_id,
            repo_type=args.repo_type,
            ignore_patterns=ignore_patterns,
            commit_message="feat: deploy Pusher-v5 PPO model, telemetry cockpit, and video gallery"
        )
        print("  --> Upload completed successfully!")
    except Exception as e:
        print(f"[ERROR] File upload failed: {e}")
        sys.exit(1)

    # Success Summary
    print("\n" + "=" * 65)
    print(" 🚀 DEPLOYMENT COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    print(f" • Hugging Face URL: https://huggingface.co/{repo_id}")
    if args.repo_type == "space":
        print(f" • Live Web App URL: https://huggingface.co/spaces/{repo_id}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
