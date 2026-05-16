# Projeto Final de TR2 - 2026_1

# PROJETO FINAL

## Streaming Adaptativo com Controle de Banda  
### Adaptive Bitrate (ABR) sobre HTTP

**Disciplina:** Teleinformática e Redes 2 (TR2)  
**Grupos:** 3 alunos

---

# 1. Contextualização

O streaming de vídeo adaptativo é hoje a forma dominante de distribuição de conteúdo multimídia na Internet. Plataformas como YouTube, Netflix e Prime Video, dentre outras, utilizam variantes do protocolo DASH (*Dynamic Adaptive Streaming over HTTP*) para ajustar dinamicamente a qualidade do vídeo de acordo com as condições da rede.

Neste projeto, o grupo irá implementar um sistema de streaming adaptativo em Python, com um servidor HTTP com controle programático de banda e variação de atraso (*jitter*), e um cliente com algoritmos de seleção adaptativa de qualidade (ABR). O sistema suporta dois servidores com failover automático e gestão de buffer com estimativa de *continuous play*.

---

## Por que este projeto?

Serviços como Netflix não transmitem vídeo continuamente: enviam segmentos de 2–4s via HTTP sobre TCP, e o cliente decide a qualidade de cada segmento com base no que observa da rede.

Entender esse mecanismo exige dominar:

- HTTP
- TCP
- Medição de vazão
- Gestão de buffer
- Políticas de decisão adaptativa

---

# 2. Objetivos de Aprendizagem

Ao concluir este projeto, o grupo deverá ser capaz de:

- Descrever o funcionamento do HTTP em modo chunked e sua relação com o TCP
- Implementar um servidor HTTP com controle de taxa de transferência programático
- Medir vazão e variação de atraso (*jitter*) de uma conexão TCP
- Implementar gestão de buffer com estimativa de *continuous play*
- Projetar e implementar pelo menos três algoritmos ABR distintos
- Implementar failover automático entre servidores
- Correlacionar eventos da aplicação com comportamentos TCP visíveis no Wireshark

---

# 3. Descrição do Sistema

## 3.1 Arquitetura

| Componente | Responsabilidade | Quem implementa |
|---|---|---|
| Servidor A (principal) | Expor manifest, servir segmentos com controle de banda e jitter | Infraestrutura |
| Servidor B (fallback) | Igual ao servidor A, rodando em porta diferente | Infraestrutura |
| Cliente | Download, ABR, buffer, failover e métricas | Grupo |

---

## 3.2 Manifest v2.0

O manifest é um JSON retornado pelo servidor ao cliente na inicialização.

| Qualidade | Bitrate (kbps) | Segmento | Cenário |
|---|---|---|---|
| 240p | 200 | ~25 KB | Conexão limitada |
| 360p | 400 | ~50 KB | Conexão ruim |
| 480p | 600 | ~75 KB | Conexão média |
| 720p | 1000 | ~125 KB | Conexão boa |
| 1080p | 1200 | ~150 KB | Conexão ótima |

---

# 4. Tarefas Obrigatórias

# Tarefa 1 — Baseline (Rate-Based ABR)

Implementar:

- Parser do manifest JSON
- Medição de vazão
- Rate-Based ABR
- BufferManager
- Registro em CSV
- Gráficos de vazão e qualidade

### Fórmula do buffer

```python
buffer += segment_duration
buffer -= tempo_real_decorrido
```

### Métricas no CSV

- Vazão
- Jitter
- Buffer
- Qualidade
- Rebuffering

---

# Tarefa 2 — Política 2 + Failover

A Política 2 deve:

- Identificar deficiências do baseline
- Implementar melhorias
- Comparar resultados

### Failover

- Detectar falha
- Verificar `/health`
- Migrar para servidor alternativo
- Registrar evento no CSV

---

# Tarefa 3 — Política 3 Estatística/Heurística

A terceira política deve incluir:

- Componente estatístico ou heurístico
- Tratamento de jitter
- Comparação com baseline

### Exemplos aceitos

- EWMA
- Detecção de tendência
- Desvio padrão
- Política híbrida

---

# Tarefa 4 — Wireshark

- Capturar tráfego TCP
- Correlacionar TCP com CSV
- Identificar failover
- Explicar eventos no relatório

---

# 5. Cronograma

| Entrega | Semana | Duração |
|---|---|---|
| Entrega 1 | 4 | 10 min |
| Entrega 2 | 7 | 10 min |
| Final | 10 | 20–25 min |

---

# 6. Relatório Final

## Estrutura obrigatória

1. Introdução
2. Arquitetura
3. Baseline
4. Deficiências
5. Política 2
6. Política 3
7. Failover
8. Wireshark
9. Discussão
10. Conclusão

---

## Gráficos obrigatórios

- Vazão + qualidade
- Buffer
- EWMA de jitter
- Comparação das 3 políticas
- Evento de failover

---

# 7. Critérios de Avaliação

| Critério | Peso |
|---|---|
| Baseline + Buffer | 15% |
| Política 2 + Failover | 20% |
| Política 3 | 20% |
| Wireshark | 15% |
| Relatório | 15% |
| Apresentação | 15% |

---

# 8. Infraestrutura

## Requisitos

- Python 3.6+
- Wireshark
- matplotlib

---

## Servidores

### Servidor A

```txt
http://137.131.178.229:8080
```

### Servidor B

```txt
http://137.131.178.229:8081
```

### Manifest

```txt
http://137.131.178.229:8080/manifest
```

---

# 8.3 Métricas CSV

| Campo | Descrição |
|---|---|
| segment | Número do segmento |
| timestamp | Horário |
| server_id | Servidor |
| quality | Qualidade |
| bitrate_kbps | Bitrate |
| throughput_kbps | Vazão |
| jitter_ms | Jitter |
| buffer_level_s | Buffer |
| rebuffer_event | Rebuffer |
| failover_total | Total de failovers |

---

# 9. Dicas

## Dica 1

Comece pelo baseline.

## Dica 2

O CSV é a base do projeto.

## Dica 3

Wireshark e CSV devem estar sincronizados.

## Dica 4

Compare com dados reais.

---

# 10. Integridade Acadêmica

O uso de IA é permitido desde que:

- O grupo compreenda o código
- As análises sejam reais
- Os dados sejam da apresentação real
