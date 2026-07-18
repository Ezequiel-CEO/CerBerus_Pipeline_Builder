# Gerenciador de Memória

## Visão Geral

O módulo `memory_manager.py` implementa um gerenciador de memória completo para otimizar o uso de recursos em processamento de dados intensivos. O gerenciador controla automaticamente o uso de RAM e VRAM, permitindo o processamento eficiente de grandes volumes de dados sem esgotar os recursos disponíveis.

## Funcionalidades

- **Detecção automática de dispositivos**: Identifica CUDA, GPU e configura o dispositivo apropriado
- **Monitoramento de recursos**: Rastreia uso de RAM e VRAM em tempo real
- **Otimização de processamento**:
  - Ajuste dinâmico de tamanho de lotes
  - Decisão inteligente de processamento em streaming
  - Carregamento sob demanda de modelos
- **Gerenciamento avançado de memória**:
  - Cache de embeddings e resultados intermediários
  - Limpeza automática quando limites são excedidos
  - Offloading de modelos entre CPU e GPU

## Uso Básico

```python
from cerberus_api.pipeline_builder.utils.memory_manager import memory_manager

# Obter uso atual de memória
memory_info = memory_manager.get_memory_usage()
print(f"RAM: {memory_info['ram_used_mb']}MB / {memory_info['ram_total_mb']}MB")

# Otimizar tamanho de lote para um conjunto de dados
batch_size = memory_manager.optimize_batch_size(data_size=1000)

# Decidir se deve usar processamento em streaming
use_streaming = memory_manager.should_process_in_streaming(
    total_data_size=500_000_000,  # 500MB
    item_sizes=[1000000, 2000000, 500000]  # Tamanhos individuais
)

# Cache de embeddings
memory_manager.cache_embedding("document_123", embedding_vector)
cached_embedding = memory_manager.get_cached_embedding("document_123")

# Limpeza de memória
memory_manager.clear_memory()
```

## Integração com Analisadores

O gerenciador de memória é utilizado pelo `EnhancedAnalyzer` para:

1. Otimizar tamanhos de lote com base na memória disponível
2. Tomar decisões sobre processamento em streaming
3. Gerenciar cache de embeddings
4. Carregar e descarregar modelos conforme necessário

## Configuração

O gerenciador possui diversos parâmetros configuráveis:

- `ram_threshold`: Limiar para limpeza automática de RAM (padrão: 0.85)
- `vram_threshold`: Limiar para offloading de GPU para CPU (padrão: 0.80)
- `enable_cuda`: Habilitar uso de CUDA quando disponível
- `enable_streaming`: Habilitar decisão de processamento em streaming
- `max_batch_size`: Tamanho máximo de lote
- `log_interval`: Intervalo para registrar uso de memória

## Requisitos

- `psutil`: Para monitoramento de RAM
- `torch`: Opcional, para suporte a CUDA/GPU 