"""Seleção de servidores por prioridade com suporte a failover."""

from domain.manifest import ServerInfo


class PriorityServerSelector:
    """Mantém o servidor ativo e troca para o próximo em caso de falha."""

    def __init__(self, servers: list[ServerInfo]) -> None:
        """
        Ordena os servidores por prioridade e seleciona o primeiro.

        Raises:
            ValueError: Se a lista de servidores estiver vazia.
        """

        if not servers:
            raise ValueError("Lista de servidores vazia")

        self.servers = sorted(servers, key=lambda server: server.priority)
        self.current_index = 0
        self.failover_total = 0

    def get_current_server(self) -> ServerInfo:
        """Retorna o servidor selecionado atualmente."""

        return self.servers[self.current_index]

    def mark_failed_and_switch(self) -> ServerInfo:
        """
        Registra falha no servidor atual e troca para o próximo fallback.

        Raises:
            RuntimeError: Se não houver outro servidor disponível.
        """

        if self.current_index + 1 >= len(self.servers):
            raise RuntimeError("Nenhum servidor de fallback disponível")

        self.current_index += 1
        self.failover_total += 1

        return self.get_current_server()
