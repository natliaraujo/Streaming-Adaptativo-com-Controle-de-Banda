"""
Define as estruturas de dados que representam o Manifest v2.0 do sistema.

O manifest descreve os servidores disponíveis, suas prioridades e parâmetros
de rede declarados, além das representações de qualidade disponíveis para os
segmentos de vídeo.

Este módulo contém apenas modelos de domínio imutáveis, sem lógica de rede,
download ou decisão de política.
"""

from dataclasses import dataclass


def normalize_server_id(server_id: str) -> str:
    """Converte aliases do manifest para os identificadores canônicos A/B."""
    prefix, separator, suffix = server_id.partition("-")
    if separator and prefix.casefold() == "srv" and suffix.casefold() in {"a", "b"}:
        return suffix.upper()
    return server_id


@dataclass(frozen=True)
class ServerInfo:
    """Metadados de um servidor disponível para baixar segmentos."""

    id: str
    """Identificador textual do servidor."""

    url: str
    """URL base usada para fazer as requisições HTTP."""

    priority: int
    """Prioridade de uso; valores menores são tentados primeiro."""

    bandwidth_kbps: float | None
    """Capacidade nominal informada pelo manifesto, quando disponível."""

    jitter_ms: float | None
    """Jitter nominal informado pelo manifesto, quando disponível."""


@dataclass(frozen=True)
class Representation:
    """Uma qualidade de vídeo disponível no manifesto."""

    quality: str
    """Rótulo textual da qualidade, como `240p`, `480p` ou `720p`."""

    bitrate_kbps: int
    """Bitrate nominal da representação em kilobits por segundo."""

    segment_bytes: int
    """Tamanho esperado de cada segmento dessa representação em bytes."""

    url_path: str
    """Caminho HTTP relativo usado para baixar segmentos dessa representação."""


@dataclass(frozen=True)
class Manifest:
    """Manifesto completo consumido pelo cliente de streaming."""

    version: str
    """Versão do formato ou conteúdo do manifesto."""

    segment_duration_s: float
    """Duração nominal de cada segmento em segundos."""

    servers: list[ServerInfo]
    """Servidores disponíveis, ordenados por prioridade após o parsing."""

    representations: list[Representation]
    """Representações disponíveis, ordenadas por bitrate após o parsing."""
