"""
Carrega e interpreta o manifest remoto do experimento.

Este módulo é responsável por realizar a requisição HTTP ao endpoint de manifest,
decodificar o JSON recebido e convertê-lo para as estruturas tipadas definidas
em `domain.manifest`.

A validação básica do formato do manifest também deve ser feita aqui, para que
erros de entrada sejam detectados antes da execução do experimento.
"""

import json
import time
import urllib.error
import urllib.request
from typing import Any

from domain.manifest import Manifest, ServerInfo, Representation
from config import HTTP_TIMEOUT_S, NETWORK_RETRY_DELAY_S


def parse_manifest(data: dict[str, Any]) -> Manifest:
    """
    Converte um dicionário de manifesto em um `Manifest` tipado.

    Servidores são ordenados por prioridade e representações por bitrate para
    simplificar a lógica de failover e seleção ABR.

    Args:
        data: Objeto JSON já decodificado em dicionário.

    Returns:
        Manifesto tipado, com servidores ordenados por prioridade e
        representações ordenadas por bitrate.

    Raises:
        ValueError: Se algum campo obrigatório não existir.
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


def load_manifest(url: str, max_attempts: int = 3) -> Manifest:
    """Baixa e interpreta o manifesto com tolerância a falhas transitórias.

    Cada tentativa usa o timeout HTTP global. Entre falhas de conexão, o cliente
    aguarda ``NETWORK_RETRY_DELAY_S`` antes de tentar novamente.

    Args:
        url: URL completa do endpoint de manifesto.
        max_attempts: Número máximo de tentativas antes de desistir.

    Returns:
        Manifesto remoto convertido para o modelo de domínio.

    Raises:
        ValueError: Se ``max_attempts`` for menor que um ou o conteúdo recebido
            não respeitar o formato esperado.
        RuntimeError: Se todas as tentativas de rede falharem.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts deve ser maior que zero.")

    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return parse_manifest(data)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Falha ao carregar manifest após {max_attempts} tentativas."
                ) from exc

            print(
                f"Falha ao carregar manifest ({attempt}/{max_attempts}): {exc}. "
                f"Nova tentativa em {NETWORK_RETRY_DELAY_S:.1f}s..."
            )
            time.sleep(NETWORK_RETRY_DELAY_S)

    raise RuntimeError("Fluxo inesperado ao carregar manifest.")
