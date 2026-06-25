# Streaming Adaptativo com Controle de Banda

Cliente Python para o projeto final de Teleinformática e Redes 2 (TR2), focado em streaming adaptativo sobre HTTP. O cliente baixa um manifesto, escolhe a qualidade dos segmentos com uma política ABR, mede métricas de rede e salva os resultados em CSV para análise.

As especificações completas do trabalho estão em [`especificações.md`](especificações.md).

## Estado Atual

Implementado:

- Parser do manifesto JSON.
- Modelo de domínio para servidores, representações, manifesto e métricas.
- Política 1 rate-based com EWMA de vazão.
- Política 2 buffer-aware com servidor secundário exclusivo para failover.
- Política 3 preditiva baseada em RNN.
- Download HTTP de segmentos.
- Medição de vazão por segmento.
- Estimativa de jitter a partir dos intervalos entre chunks.
- Média móvel exponencial (EWMA) do jitter.
- Buffer finito com enchimento inicial, alvo, teto e detecção de rebuffering.
- Escrita de métricas em CSV.
- Geração de gráfico de vazão medida e qualidade escolhida.
- Failover automático com verificação de `/health` e métricas de duração.

Etapas externas ao código ainda dependem da execução dos experimentos, captura
no Wireshark e elaboração do relatório final.

## Dependências

Requisitos principais:

- Python 3.10+ recomendado.
- `matplotlib` para gerar gráficos.
- `torch` para treinar e executar a Política 3.
- Wireshark para a etapa de captura e análise TCP.

As Políticas 1 e 2 usam apenas a biblioteca padrão para rede e persistência. A
Política 3 depende de PyTorch.

Instalação sugerida:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Configuração

Os parâmetros principais ficam em [`config.py`](config.py):

```python
MANIFEST_URL = "http://137.131.178.229:8080/manifest"
NUM_SEGMENTS = 100
BUFFER_MAX_S = 30.0
BUFFER_TARGET_S = 15.0
BUFFER_MIN_S = 4.0
BUFFER_CRITICAL_S = 1.0
ABR_HISTORY_SIZE = 5
SAFETY_FACTOR = 0.92
ALPHA_EWMA = 0.3
HTTP_TIMEOUT_S = 15
HEALTH_CHECK_TIMEOUT_S = 2.0
NETWORK_RETRY_DELAY_S = 2.0
NETWORK_RECOVERY_MAX_WAIT_S = 120.0
```

O manifesto configurado aponta para o servidor principal da infraestrutura do projeto. A especificação também prevê um servidor fallback em:

```text
http://137.131.178.229:8081
```

## Como Executar

Para rodar as Políticas 1 e 2:

```bash
python -m scripts.run_policy1
python -m scripts.run_policy2
```

Para comparar seus resultados:

```bash
python -m scripts.compare_policies
```

Para gerar o gráfico de vazão e qualidade:

```bash
python3 scripts/plot_baseline.py
```

Para gerar os gráficos individuais e as quatro comparações finais:

```bash
python scripts/generate_final_plots.py
```

Os CSVs esperados são `outputs/metricas_policy1.csv`,
`outputs/metricas_policy2.csv` e `outputs/metricas_policy3_rnn.csv`. As figuras
são salvas em `outputs/figures/`.

Para executar e plotar o experimento controlado de failover da Política 2:

```bash
python scripts/run_policy2_failover_experiment.py
python scripts/plot_policy2_failover_experiment.py
```

Por padrão, o servidor `A` fica indisponível apenas dos segmentos 9 a 12. A
tentativa do segmento 9 aciona o fallback e registra `failover_event = 1` no
CSV; a partir do segmento 13, o runner revalida o `/health` de `A` e pode voltar
ao servidor principal.

Também é possível ajustar a janela de falha:

```bash
python scripts/run_policy2_failover_experiment.py --fail-after-segment 8 --fail-duration-segments 4
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
├── analysis/             # Leitura de métricas e geração de gráficos
├── domain/               # Modelos de manifesto e métricas
├── experiment/           # Runner do experimento e escrita CSV
├── failover/             # Seleção de servidores por prioridade
├── network/              # Cliente de manifesto e downloader de segmentos
├── player/               # Gerenciamento de buffer
├── policy/               # Políticas 1, 2 e 3
├── monitoring/           # Probes e observações para a RNN
├── models/               # Modelo, dataset e treinamento da RNN
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
| `throughput_kbps` | Vazão medida durante o download |
| `throughput_ewma_kbps` | EWMA da vazão medida |
| `download_time_s` | Tempo total de download |
| `jitter_network_ms` | Jitter medido entre chunks HTTP |
| `jitter_ewma_ms` | EWMA do jitter |
| `buffer_level_s` | Nível do buffer após baixar o segmento |
| `buffer_can_play` | Indica se havia buffer suficiente para reprodução contínua |
| `rebuffer_event` | Indica ocorrência de rebuffering |
| `stall_duration_s` | Duração acumulada do stall |
| `playback_wait_s` | Espera para abrir espaço quando o buffer atinge o máximo |
| `failover_event` | Indica troca de servidor no segmento |
| `failover_duration_s` | Tempo gasto para confirmar o alternativo |
| `failover_total` | Total acumulado de failovers |

## Política Baseline

A Política 1 está em [`policy/rate_based.py`](policy/rate_based.py). Ela aplica
um fator de segurança à EWMA das vazões medidas:

```text
vazao_segura = vazao_ewma * SAFETY_FACTOR
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

O runner marca servidores que falharam, confirma o alternativo por meio de
`/health` e registra evento, duração e total acumulado de failovers no CSV.

## Arquivos Gerados

Arquivos em `outputs/`, caches Python, `.venv/` e capturas Wireshark são ignorados pelo Git. Para compartilhar resultados finais, gere os arquivos localmente ou adicione exceções específicas no `.gitignore`.

## Próximos Passos

1. Rodar os experimentos das políticas sob as mesmas condições.
2. Gerar e interpretar o gráfico comparativo.
3. Capturar tráfego no Wireshark e correlacionar os eventos TCP com o CSV.
4. Consolidar resultados e conclusões no relatório.
