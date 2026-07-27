"""Helpers for downloading and uploading ML artifacts from/to Azure Blob Storage."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_BLOB_HOST_SUFFIX = ".blob.core.windows.net"


def parse_blob_uri(uri: str) -> tuple[str, str, str]:
    """Parse https://<account>.blob.core.windows.net/<container>/<blob> into
    (account, container, blob_path)."""
    parsed = urlparse(uri)
    if parsed.scheme not in ("https", "http") or _BLOB_HOST_SUFFIX not in (parsed.netloc or ""):
        raise ValueError(
            f"Invalid Azure Blob URI: {uri!r}. "
            f"Expected https://<account>.blob.core.windows.net/<container>/<blob>"
        )
    account = parsed.netloc.replace(_BLOB_HOST_SUFFIX, "")
    parts = parsed.path.lstrip("/").split("/", 1)
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"Azure Blob URI missing container or blob path: {uri!r}")
    container, blob_path = parts[0], parts[1]
    return account, container, blob_path


def _get_blob_client(account: str, container: str, blob_path: str):
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobClient

    credential = DefaultAzureCredential()
    url = f"https://{account}{_BLOB_HOST_SUFFIX}/{container}/{blob_path}"
    return BlobClient.from_blob_url(url, credential=credential)


def download_blob_uri(uri: str, local_path: str, **_) -> str:
    """Download an Azure Blob to local_path. Returns the local path."""
    account, container, blob_path = parse_blob_uri(uri)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", uri, local_path)
    client = _get_blob_client(account, container, blob_path)
    with open(local_path, "wb") as f:
        stream = client.download_blob()
        stream.readinto(f)
    return local_path


def upload_file(local_path: str, uri: str, **_) -> str:
    """Upload a local file to an Azure Blob URI. Returns the URI."""
    account, container, blob_path = parse_blob_uri(uri)
    logger.info("Uploading %s -> %s", local_path, uri)
    client = _get_blob_client(account, container, blob_path)
    with open(local_path, "rb") as f:
        client.upload_blob(f, overwrite=True)
    return uri


def blob_exists(uri: str, **_) -> bool:
    """Return True if the Azure Blob exists."""
    account, container, blob_path = parse_blob_uri(uri)
    client = _get_blob_client(account, container, blob_path)
    return client.exists()
