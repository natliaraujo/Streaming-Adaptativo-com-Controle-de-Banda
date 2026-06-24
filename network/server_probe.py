"""
Implementa probes leves para avaliar o estado dos servidores.

Este módulo é usado por políticas que precisam comparar servidores antes de
baixar o próximo segmento. A função principal, `probe_server_health`, mede se o
servidor está acessível e estima sua latência por meio de uma requisição HTTP
leve ao endpoint `/health`.

Em uma versão posterior, este módulo também pode medir vazão aproximada baixando
uma pequena parte de um segmento.
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from domain import ServerInfo


@dataclass(frozen=True)
class ServerProbeResult:
    """
    Resultado de um probe realizado em um servidor.

    Attributes:
        server_id: Identificador do servidor testado.
        ok: Indica se o servidor respondeu com sucesso.
        latency_ms: Tempo de resposta medido em milissegundos.
        throughput_kbps: Vazão estimada em kbps. Na versão health-check simples,
            este valor fica como None.
        error: Mensagem de erro, caso o probe tenha falhado.
    """

    server_id: str
    ok: bool
    latency_ms: float
    throughput_kbps: float | None
    error: str | None = None


def probe_server_health(
    server: ServerInfo,
    timeout_s: float = 1.0,
) -> ServerProbeResult:
    """
    Verifica se um servidor está saudável usando uma requisição HTTP leve.

    A função tenta acessar o endpoint `/health` do servidor. Se a resposta tiver
    status HTTP 2xx, o servidor é considerado saudável. A latência é estimada
    como o tempo total da requisição.

    Args:
        server: Servidor que será testado.
        timeout_s: Tempo máximo de espera pela resposta, em segundos.

    Returns:
        Um `ServerProbeResult` contendo disponibilidade, latência e possível
        mensagem de erro.
    """
    url: str = f"{server.url.rstrip('/')}/health"
    start_s: float = time.time()

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            latency_ms: float = (time.time() - start_s) * 1000.0
            ok: bool = response.status == 200

            body: bytes = response.read(4096)
            if ok and body:
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    payload = None

                if isinstance(payload, dict) and "status" in payload:
                    ok = str(payload["status"]).lower() == "ok"

            return ServerProbeResult(
                server_id=server.id,
                ok=ok,
                latency_ms=latency_ms,
                throughput_kbps=None,
                error=None if ok else "Health check não retornou status ok.",
            )

    except urllib.error.URLError as exc:
        latency_ms = (time.time() - start_s) * 1000.0

        return ServerProbeResult(
            server_id=server.id,
            ok=False,
            latency_ms=latency_ms,
            throughput_kbps=None,
            error=str(exc),
        )

    except TimeoutError as exc:
        latency_ms = (time.time() - start_s) * 1000.0

        return ServerProbeResult(
            server_id=server.id,
            ok=False,
            latency_ms=latency_ms,
            throughput_kbps=None,
            error=str(exc),
        )
