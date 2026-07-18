# Analisadores com Gerenciamento de Memória

## Visão Geral

Este diretório contém os analisadores do CerBerus FMK, incluindo o `EnhancedAnalyzer` integrado com o gerenciador de memória para processamento eficiente de grandes volumes de texto.

## EnhancedAnalyzer

O `EnhancedAnalyzer` estende o `ContentAnalyzer` base com funcionalidades avançadas e otimizações de memória:

### Principais recursos

- **Processamento otimizado**:
  - Análise em lotes com tamanho adaptativo
  - Processamento em streaming para grandes conjuntos de dados
  - Carregamento de modelos sob demanda

- **Características avançadas**:
  - Geração de embeddings para análise semântica
  - Cálculo de similaridade entre documentos
  - Extração aprimorada de metadados e tópicos
  - Suporte para summarização de texto

- **Integração com gerenciador de memória**:
  - Monitoramento automático de recursos
  - Otimização de uso de RAM/VRAM
  - Cache inteligente de resultados intermediários

## Uso Básico

```python
from cerberus_api.pipeline_builder.analyzers.enhanced_analyzer import EnhancedAnalyzer
from cerberus_api.pipeline_builder.models import ContentChunk

# Criar analisador com configuração para eficiência de memória
analyzer = EnhancedAnalyzer(
    storage_dir="output/results",
    stream_processing=True,     # Ativar processamento em streaming
    memory_efficient=True,      # Usar otimizações de memória
    batch_size=10,              # Tamanho inicial de lote (será otimizado)
    use_gpu=True                # Usar GPU quando disponível
)

# Analisar chunks de conteúdo
results = analyzer.analyze(chunks)
```

## Estratégias de Processamento

O `EnhancedAnalyzer` implementa diversas estratégias para processamento eficiente:

### 1. Processamento em Lotes

Usado para volumes médios de dados, divide o processamento em lotes com tamanho otimizado automaticamente:

```python
results = analyzer._process_in_batches(chunks)
```

### 2. Processamento em Streaming

Usado para grandes volumes de dados, processa cada chunk individualmente, economizando memória:

```python
results = analyzer._process_in_streaming(chunks)
```

### 3. Processamento de Chunks Grandes

Técnica especial para processar chunks individuais muito grandes:

```python
result = analyzer._process_large_chunk_in_streaming(chunk)
```

## Configuração

O `EnhancedAnalyzer` possui diversos parâmetros configuráveis:

```python
analyzer = EnhancedAnalyzer(
    # Parâmetros de relevância
    min_relevance_score=0.5,
    keywords_ratio=0.05,
    
    # Configuração de idioma
    stopwords_langs=["portuguese", "english"],
    
    # Configuração de armazenamento
    storage_dir="output/results",
    
    # Configuração de processamento
    batch_size=10,
    max_text_length=50000,
    stream_processing=True,
    memory_efficient=True,
    
    # Configuração de modelos
    use_embeddings=True,
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    spacy_model="pt_core_news_sm",
    
    # Controle de GPU
    use_gpu=True
)
``` 