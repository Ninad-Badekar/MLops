"""Sync trained artifacts to Azure Blob Storage for serving."""

from __future__ import annotations

import argparse
import os
import sys

from src.aws_artifacts import upload_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload serving artifacts to Azure Blob Storage")
    parser.add_argument(
        "--model",
        default="models/model.pkl",
        help="Local model path",
    )
    parser.add_argument(
        "--baseline",
        default="data/processed/train_processed_mean.csv",
        help="Local baseline training CSV path",
    )
    parser.add_argument(
        "--storage-account",
        default=os.getenv("AZURE_STORAGE_ACCOUNT"),
        dest="storage_account",
        help="Azure storage account name (or set AZURE_STORAGE_ACCOUNT)",
    )
    parser.add_argument(
        "--container",
        default=os.getenv("AZURE_STORAGE_CONTAINER", "artifacts"),
        help="Blob container name (default: artifacts)",
    )
    parser.add_argument(
        "--prefix",
        default="serving",
        help="Blob path prefix",
    )
    args = parser.parse_args()

    if not args.storage_account:
        print("ERROR: --storage-account or AZURE_STORAGE_ACCOUNT is required", file=sys.stderr)
        return 1

    base = f"https://{args.storage_account}.blob.core.windows.net/{args.container}"

    model_uri = f"{base}/{args.prefix}/model.pkl"
    upload_file(args.model, model_uri)
    print(f"Uploaded model -> {model_uri}")

    if os.path.exists(args.baseline):
        baseline_uri = f"{base}/{args.prefix}/train_processed_mean.csv"
        upload_file(args.baseline, baseline_uri)
        print(f"Uploaded baseline -> {baseline_uri}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
