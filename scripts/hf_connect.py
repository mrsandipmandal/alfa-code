"""Connect to HuggingFace Hub: login, whoami, create dataset repo, upload."""
import argparse, os, sys
from pathlib import Path

def get_token(args):
    tok = args.token or os.getenv("HF_TOKEN")
    if not tok and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                tok = line.split("=", 1)[1].strip()
    return tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=None, help="hf_xxx token")
    ap.add_argument("--username", default=None)
    ap.add_argument("--repo", default="alfa-code-multimodal")
    ap.add_argument("--create-repo", action="store_true")
    ap.add_argument("--upload", default=None, help="local path to upload, e.g. data/processed")
    args = ap.parse_args()

    from huggingface_hub import HfApi, login, whoami
    tok = get_token(args)
    if not tok:
        print("No HF_TOKEN found. Create one at https://huggingface.co/settings/tokens (read+write), then:")
        print("  python scripts/hf_connect.py --token hf_xxx")
        print("  # or copy .env.example to .env and fill HF_TOKEN")
        sys.exit(1)

    login(token=tok)
    info = whoami(token=tok)
    print(f"Logged in as: {info['name']}")

    if args.create_repo:
        api = HfApi()
        repo_id = f"{info['name']}/{args.repo}" if "/" not in args.repo else args.repo
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"Dataset repo ready: https://huggingface.co/datasets/{repo_id}")

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        username = args.username or info["name"]
        repo_id = f"{username}/{args.repo}" if "/" not in args.repo else args.repo
        api.upload_folder(folder_path=args.upload, repo_id=repo_id, repo_type="dataset")
        print(f"Uploaded {args.upload} -> {repo_id}")

if __name__ == "__main__":
    main()
