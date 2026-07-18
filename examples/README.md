# Gerenciamento de Memória no CerBerus FMK

Este diretório contém exemplos de uso do gerenciador de memória integrado ao framework CerBerus.

## Exemplo: `memory_management_example.py`

Este exemplo demonstra como o sistema processa grandes volumes de texto de forma eficiente, utilizando o gerenciador de memória para otimizar o uso de recursos.

### Funcionalidades demonstradas:

1. **Gerenciamento automático de recursos**:
   - Detecção de GPU/CUDA
   - Monitoramento de RAM/VRAM
   - Limpeza automática de memória

2. **Processamento adaptativo**:
   - Tamanho de lotes ajustados dinamicamente
   - Processamento em streaming para grandes volumes
   - Decisão automática baseada na disponibilidade de recursos

3. **Otimizações de memória**:
   - Carregamento sob demanda de modelos
   - Cache de embeddings
   - Liberação de memória entre processamentos

### Como executar:

```bash
python -m cerberus_api.pipeline_builder.examples.memory_management_example
```

### Saída esperada:

O exemplo processa três conjuntos de dados de diferentes tamanhos:
- Pequeno: ~10 chunks
- Médio: ~50 chunks 
- Grande: ~200 chunks

Para cada conjunto, são exibidas estatísticas de uso de memória, tempo de processamento e métricas de relevância.

### Requisitos opcionais:

Para aproveitar todas as funcionalidades:

```bash
pip install spacy sentence-transformers
python -m spacy download pt_core_news_sm
```

## Implementação do Gerenciador de Memória

O gerenciador de memória (`MemoryManager`) está implementado em `cerberus_api/pipeline_builder/utils/memory_manager.py` e oferece:

- Interface única para monitoramento de recursos
- Métodos para otimização de lotes
- Funções para limpeza de memória
- Cache de objetos pesados
- Decisão inteligente para streaming

## Integração com Analisadores

O `EnhancedAnalyzer` utiliza o gerenciador de memória para:

1. Determinar o tamanho ideal de lotes
2. Decidir quando usar processamento em streaming
3. Gerenciar cache de embeddings
4. Carregar modelos sob demanda
5. Liberar recursos após processamento 