# Streaming Adaptativo com Controle de Banda

Cliente Python para o projeto final de Teleinformática e Redes 2 (TR2), focado em streaming adaptativo sobre HTTP. O cliente baixa um manifesto, escolhe a qualidade dos segmentos com uma política ABR, mede métricas de rede e salva os resultados em CSV para análise.

As especificações completas do trabalho estão em [`especificações.md`](especificações.md).

## Estado Atual

Implementado:

- Parser do manifesto JSON.
- Modelo de domínio para servidores, representações, manifesto e métricas.
- Política baseline `RateBasedAbrPolicy`.
- Download HTTP de segmentos.
- Medição de vazão por segmento.
- Estimativa de jitter a partir dos intervalos entre chunks.
- Média móvel exponencial (EWMA) do jitter.
- Simulação simples de buffer e detecção de rebuffering.
- Escrita de métricas em CSV.
- Geração de gráfico de vazão medida e qualidade escolhida.
- Seleção de servidor por prioridade e tentativa de failover em erro de download.

Ainda falta ou está parcial:

- Política 2 melhorada, sensível a buffer, jitter ou outra heurística.
- Política 3 estatística/heurística.
- Verificação explícita de `/health` antes do failover.
- Gráficos obrigatórios adicionais: buffer, EWMA de jitter, comparação das três políticas e evento de failover.
- Experimentos comparativos entre políticas.
- Captura e análise no Wireshark.
- Relatório final.

## Dependências

Requisitos principais:

- Python 3.10+ recomendado.
- `matplotlib` para gerar gráficos.
- Wireshark para a etapa de captura e análise TCP.

O cliente baseline usa apenas a biblioteca padrão do Python para rede, CSV e parsing JSON. A dependência externa atual é o `matplotlib`, usado por `analysis/plots.py`.

Instalação sugerida:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install matplotlib
```

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install matplotlib
```

## Configuração

Os parâmetros principais ficam em [`config.py`](config.py):

```python
MANIFEST_URL = "http://137.131.178.229:8080/manifest"
NUM_SEGMENTS = 10
ABR_HISTORY_SIZE = 5
SAFETY_FACTOR = 0.8
ALPHA_EWMA = 0.3
HTTP_TIMEOUT_S = 10
```

O manifesto configurado aponta para o servidor principal da infraestrutura do projeto. A especificação também prevê um servidor fallback em:

```text
http://137.131.178.229:8081
```

## Como Executar

Para rodar o experimento baseline:

```bash
python3 main.py
```

O resultado é salvo em:

```text
outputs/metricas_baseline.csv
```

Para gerar o gráfico de vazão e qualidade:

```bash
python3 scripts/plot_baseline.py
```

Por padrão, o gráfico é salvo em:

```text
outputs/plots/grafico_vazao_qualidade.png
```

Também é possível informar caminhos manualmente:

```bash
python3 scripts/plot_baseline.py --csv outputs/metricas_baseline.csv --output outputs/plots/grafico_vazao_qualidade.png
```

## Estrutura do Projeto

```text
.
├── abr/                  # Políticas ABR
├── analysis/             # Leitura de métricas e geração de gráficos
├── domain/               # Modelos de manifesto e métricas
├── experiment/           # Runner do experimento e escrita CSV
├── failover/             # Seleção de servidores por prioridade
├── network/              # Cliente de manifesto e downloader de segmentos
├── player/               # Gerenciamento de buffer
├── scripts/              # Scripts de linha de comando
├── config.py             # Constantes de configuração
├── main.py               # Entrada do experimento baseline
└── especificações.md     # Enunciado/escopo do projeto
```

## Métricas Geradas

O CSV atual contém uma linha por segmento com os campos:

| Campo | Descrição |
|---|---|
| `segment` | Número sequencial do segmento |
| `timestamp` | Horário da medição em ISO 8601 |
| `server_id` | Servidor usado no download |
| `quality` | Qualidade escolhida pela política ABR |
| `bitrate_kbps` | Bitrate nominal da representação |
| `vazao_kbps` | Vazão medida durante o download |
| `download_time_s` | Tempo total de download |
| `jitter_network_ms` | Jitter medido entre chunks HTTP |
| `jitter_ewma_ms` | EWMA do jitter |
| `buffer_level_s` | Nível do buffer após baixar o segmento |
| `buffer_can_play` | Indica se havia buffer suficiente para reprodução contínua |
| `rebuffer_event` | Indica ocorrência de rebuffering |
| `stall_duration_s` | Duração acumulada do stall |
| `failover_total` | Total acumulado de failovers |

## Política Baseline

A política baseline está em [`abr/rate_based.py`](abr/rate_based.py). Ela calcula a média das últimas vazões medidas e aplica um fator de segurança:

```text
vazao_segura = media_vazao_recente * SAFETY_FACTOR
```

Depois escolhe a maior representação cujo bitrate nominal seja menor ou igual à vazão segura. Quando ainda não há histórico, a política começa pela menor representação disponível.

## Buffer

O buffer é atualizado de forma aproximada usando tempo real decorrido durante o download:

```text
buffer -= tempo_real_decorrido
buffer += duracao_do_segmento
```

Se o consumo ultrapassa o nível disponível, o cliente registra stall e marca um evento de rebuffering.

## Failover

O projeto já tem `PriorityServerSelector`, que ordena servidores por prioridade e troca para o próximo servidor quando ocorre erro no download. No estado atual, essa troca é feita sem chamada explícita a `/health`, então a parte de failover ainda deve ser completada para atender totalmente à especificação.

## Arquivos Gerados

Arquivos em `outputs/`, caches Python, `.venv/` e capturas Wireshark são ignorados pelo Git. Para compartilhar resultados finais, gere os arquivos localmente ou adicione exceções específicas no `.gitignore`.

## Próximos Passos

1. Implementar uma política ABR sensível ao buffer em `abr/buffer_aware.py`.
2. Implementar uma terceira política estatística/heurística em `abr/rnn_policy.py` ou em novo módulo.
3. Adicionar verificação `/health` no failover.
4. Gerar gráficos de buffer, jitter EWMA, comparação de políticas e failover.
5. Rodar cenários comparativos e salvar conclusões para o relatório.
6. Capturar tráfego no Wireshark e correlacionar os eventos TCP com o CSV.
