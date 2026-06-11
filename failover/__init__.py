"""Estratégias de seleção e troca de servidores."""

from failover.server_selector import PriorityServerSelector

__all__ = [
    "PriorityServerSelector",
]
