# CerBerus AI Manager

## Visão Geral

O módulo `ai_manager` é uma extensão do CerBerus FMK que fornece funcionalidades avançadas para gerenciamento de modelos de IA, incluindo fine-tuning, inferência otimizada e aprendizado contínuo a partir de dados processados pelo PipelineBuilder.

## Arquitetura

O AI Manager é estruturado em três componentes principais:

```
ai_manager/
├── __init__.py
├── model_hub.py          # Gerenciamento e carregamento de modelos
├── fine_tuning/          # Módulos para treinamento e fine-tuning
│   ├── __init__.py
│   ├── dataset_builder.py # Preparação de datasets para treinamento
│   ├── training_manager.py # Gerenciamento de treinamento
│   └── evaluation.py     # Avaliação de modelos treinados
├── inference/            # Inferência e execução eficiente
│   ├── __init__.py
│   ├── model_server.py   # Servidor de modelos para inferência
│   ├── batch_processor.py # Processamento em lote de requisições
│   └── optimizers.py     # Otimizações para modelos (quantização, pruning, etc.)
├── adaptation/           # Adaptação e melhoria contínua
│   ├── __init__.py
│   ├── continuous_learning.py # Aprendizado contínuo
│   ├── feedback_processor.py  # Processamento de feedback
│   └── knowledge_distillation.py # Destilação de conhecimento
└── utils/                # Utilitários
    ├── __init__.py
    ├── model_metrics.py  # Métricas para avaliação de modelos
    └── config_templates.py # Templates de configuração
```

## Funcionalidades Principais

### 1. Model Hub

O `ModelHub` é o componente central que gerencia todos os modelos disponíveis no sistema:

- **Carregamento Inteligente**: Carrega e descarrega modelos sob demanda
- **Catálogo de Modelos**: Mantém informações sobre modelos disponíveis
- **Versionamento**: Controla versões de modelos e seus checkpoints
- **Compatibilidade**: Fornece interfaces uniformes para diferentes tipos de modelos

### 2. Fine-tuning

O módulo de fine-tuning permite personalizar modelos usando dados processados pelo PipelineBuilder:

- **Preparação de Datasets**: Converte chunks e análises em datasets formatados para treinamento
- **Gerenciamento de Treinamento**: Configura e executa processos de treinamento
- **Monitoramento**: Acompanha progresso, métricas e recursos durante o treinamento
- **Estratégias de Fine-tuning**: Suporte para diferentes técnicas (LoRA, QLoRA, full fine-tuning)

### 3. Inferência Otimizada

O módulo de inferência fornece mecanismos eficientes para executar modelos:

- **Servidor de Modelos**: Expõe modelos via API REST para consumo
- **Processamento em Lote**: Agrupa requisições para melhor eficiência
- **Otimizações**: Implementa técnicas como quantização, pruning e compilação para diferentes hardwares
- **Cache de Resultados**: Armazena resultados comuns para resposta instantânea

### 4. Adaptação Contínua

O módulo de adaptação permite que os modelos evoluam com o tempo:

- **Aprendizado Contínuo**: Atualiza modelos com novos dados sem retreinamento completo
- **Processamento de Feedback**: Incorpora feedback de usuários para melhorar modelos
- **Destilação de Conhecimento**: Transfere conhecimento de modelos maiores para menores

## Integração com PipelineBuilder

O AI Manager se integra ao PipelineBuilder existente:

1. **Alimentação de Dados**: Utiliza pipelines gerados para treinar e melhorar modelos
2. **Feedback de Análise**: Incorpora resultados de análise para ajustar modelos
3. **Ciclo Completo**: Implementa um ciclo de coleta → análise → treinamento → inferência

## Uso Básico

```python
from cerberus_api.pipeline_builder.ai_manager import ModelHub
from cerberus_api.pipeline_builder.ai_manager.fine_tuning import TrainingManager
from cerberus_api.pipeline_builder.ai_manager.inference import ModelServer
from cerberus_api.pipeline_builder.core import PipelineBuilder

# Inicializar o hub de modelos
model_hub = ModelHub(models_dir="models", cache_dir="cache")

# Carregar um modelo pelo nome
model = model_hub.load_model("cerberus-base-7b", device="auto")

# Preparar dados para fine-tuning a partir de um pipeline existente
pipeline_builder = PipelineBuilder()
pipeline = pipeline_builder.load_pipeline("pipeline_123")

# Configurar e iniciar fine-tuning
training_manager = TrainingManager(model_hub=model_hub)
fine_tuned_model = training_manager.fine_tune(
    base_model="cerberus-base-7b",
    pipeline=pipeline,
    output_name="cerberus-custom-7b",
    training_args={
        "epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 8,
        "use_lora": True
    }
)

# Servir modelo para inferência
model_server = ModelServer(model_hub=model_hub)
model_server.serve_model(
    model_name="cerberus-custom-7b",
    port=8000,
    max_batch_size=4
)
```

## Requisitos

- Python 3.8+
- PyTorch 2.0+
- transformers 4.30+
- datasets
- accelerate
- bitsandbytes (opcional, para quantização)
- onnxruntime (opcional, para otimização ONNX)
- triton (opcional, para otimização com Triton)

## Próximos Passos

1. **Implementação do ModelHub**: Sistema central de gerenciamento de modelos
2. **Integração com Pipelines**: Conexão com dados processados pelo PipelineBuilder
3. **Infraestrutura de Fine-tuning**: Mecanismos para treinar modelos a partir de pipelines
4. **Servidor de Inferência**: API para consumo de modelos treinados
5. **Otimizações de Inferência**: Técnicas para melhorar eficiência e latência
6. **Mecanismos de Feedback**: Sistemas para incorporar feedback e melhorar modelos 