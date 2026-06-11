"""Cliente para baixar e converter o manifesto JSON em objetos de domínio."""

import json
import urllib.request
from typing import Any

from domain.manifest import Manifest, ServerInfo, Representation


def parse_manifest(data: dict[str, Any]) -> Manifest:
    """
    Converte um dicionário de manifesto em um `Manifest` tipado.

    Servidores são ordenados por prioridade e representações por bitrate para
    simplificar a lógica de failover e seleção ABR.

    Raises:
        ValueError: Se algum campo obrigatório esperado no manifesto não existir.
    """

    try:
        servers = [
            ServerInfo(
                id=str(server["id"]),
                url=str(server["url"]),
                priority=int(server["priority"]),
                bandwidth_kbps=(
                    None
                    if server["bandwidth_kbps"] is None
                    else float(server["bandwidth_kbps"])
                ),
                jitter_ms=(
                    None
                    if server["jitter_ms"] is None
                    else float(server["jitter_ms"])
                ),
            )
            for server in data["servers"]
        ]

        representations = [
            Representation(
                quality=str(rep["quality"]),
                bitrate_kbps=int(rep["bitrate_kbps"]),
                segment_bytes=int(rep["segment_bytes"]),
                url_path=str(rep["url_path"]),
            )
            for rep in data["representations"]
        ]

        servers.sort(key=lambda server: server.priority)
        representations.sort(key=lambda rep: rep.bitrate_kbps)

        return Manifest(
            version=str(data["version"]),
            segment_duration_s=float(data["segment_duration_s"]),
            servers=servers,
            representations=representations,
        )

    except KeyError as e:
        raise ValueError(f"Campo obrigatório ausente no manifest: {e}") from e


def load_manifest(url: str) -> Manifest:
    """Baixa o manifesto JSON da URL informada e retorna o modelo tipado."""

    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return parse_manifest(data)
