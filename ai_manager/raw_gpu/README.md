# raw_gpu: Sistema de Aceleração Nativa para CerBerusFMK

Este módulo fornece uma implementação nativa para aceleração de modelos de IA usando GPU e CPU, sem dependência de frameworks externos como PyTorch, TensorFlow ou llama-cpp-python.

## Visão Geral

O sistema `raw_gpu` é projetado para otimizar a execução de modelos de linguagem no framework CerBerusFMK, oferecendo:

- Detecção automática de hardware (CPU/GPU)
- Gerenciamento otimizado de memória
- Inferência eficiente sem dependências externas
- Fallback automático entre GPU e CPU
- Suporte nativo para formatos GGUF

## Como Usar

### Exemplo Básico

```python
from cerberus_api.pipeline_builder.ai_manager.raw_gpu import create_optimized_engine

# Criar motor otimizado para o hardware disponível
engine = create_optimized_engine(
    model_path="/caminho/para/modelo.gguf",
    use_gpu=True  # Automático se disponível
)

# Gerar texto
response = engine.generate(
    prompt="Qual é a capital do Brasil?",
    max_tokens=100,
    temperature=0.7
)

print(response)
```

### Detecção de Hardware

```python
from cerberus_api.pipeline_builder.ai_manager.raw_gpu.hardware_abstraction import hardware

# Verificar disponibilidade de GPU
if hardware.is_cuda_available():
    print("GPU disponível!")
    
# Obter informações detalhadas
gpu_info = hardware.gpu_info
cpu_info = hardware.cpu_info
memory_info = hardware.get_memory_info()

print(f"GPU: {gpu_info['has_gpu']}")
print(f"VRAM Total: {memory_info['vram_total_mb']} MB")
print(f"CPU: {cpu_info['vendor']} com {cpu_info['cores']} núcleos")
```

### Gerenciamento de Memória

```python
from cerberus_api.pipeline_builder.ai_manager.raw_gpu import GPUManager

# Inicializar gerenciador
gpu_manager = GPUManager()

# Verificar memória
memory_info = gpu_manager.get_memory_info()
print(f"RAM: {memory_info['ram_used_mb']}/{memory_info['ram_total_mb']} MB")
print(f"VRAM: {memory_info['vram_used_mb']}/{memory_info['vram_total_mb']} MB")

# Otimizar tamanho de batch
batch_size = gpu_manager.optimize_batch_size(32, 10.0)  # items, MB por item
print(f"Tamanho de batch ótimo: {batch_size}")
```

## Componentes

### ModelEngine

Motor de inferência para modelos GGUF, fornecendo:
- Carregamento eficiente de modelos
- Tokenização e geração de texto
- Gerenciamento de contexto
- Streaming para textos longos

### GPUManager

Gerencia recursos de GPU/CPU:
- Transferências de memória
- Otimização de uso de VRAM/RAM
- Fallback CPU quando necessário
- Limpeza automática de memória

### HardwareAbstraction

Detecta e abstrai recursos de hardware:
- Informações detalhadas sobre CPU/GPU 
- Monitoramento de uso de memória
- Recomendações de otimização
- Detecção automática de capacidades

### CUDAKernels 

Implementações nativas de operações de tensor:
- Operações básicas (matmul, softmax)
- Funções de ativação (gelu, relu)
- Processamento de lote
- Automatização de transferências

## Requisitos

- Python 3.8+
- Sistema operacional compatível (Linux, Windows, macOS)
- GPU NVIDIA (opcional, para aceleração CUDA)
- Drivers NVIDIA atualizados (para GPU)

## Arquitetura

O sistema segue uma arquitetura em camadas:

1. **Camada de Hardware**: Detecção e abstração de hardware
2. **Camada de Memória**: Gerenciamento de memória e transferências
3. **Camada de Operações**: Implementações nativas de kernels
4. **Camada de Modelo**: Carregamento e execução de modelos
5. **API de Alto Nível**: Interface simplificada para o usuário

## Limitações Atuais

- Suporte apenas para modelos no formato GGUF
- Otimizações avançadas ainda em desenvolvimento
- Suporte limitado para quantização

## Roadmap

- [x] Implementação base do sistema
- [x] Detecção de hardware
- [x] Gerenciamento básico de memória
- [x] Integração com ModelHub
- [ ] Implementação completa de kernels CUDA
- [ ] Otimizações AVX2/AVX512 para CPU
- [ ] Implementação de streaming eficiente
- [x] Suporte para quantização INT4/INT8

## Contribuição

O desenvolvimento segue a arquitetura de componentes descrita em `implementation_plan.md`. 
Contribuições são bem-vindas conforme o roadmap.

## Licença

Este módulo é parte do CerBerusFMK e segue a mesma licença do projeto principal. 