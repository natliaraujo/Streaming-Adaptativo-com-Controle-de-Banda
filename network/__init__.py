"""Clientes de rede para manifesto e segmentos de vídeo."""

from network.manifest_client import load_manifest, parse_manifest
from network.segment_downloader import DownloadResult, download_segment

__all__ = [
    "DownloadResult",
    "download_segment",
    "load_manifest",
    "parse_manifest",
]
