# Plano de Implementação do AI Manager

Este documento descreve o plano de implementação detalhado para o módulo `ai_manager` do CerBerus FMK, que fornecerá recursos de gerenciamento, fine-tuning e inferência de modelos de IA.

## Estrutura Básica

```
ai_manager/
├── __init__.py                 ✓ (Implementado)
├── model_hub.py                ✓ (Implementado)
├── fine_tuning/
│   ├── __init__.py             ✓ (Implementado)
│   ├── dataset_builder.py      ✓ (Implementado)
│   ├── training_manager.py     ⏳ (Em seguida)
│   └── evaluation.py           🔜 (Futuro)
├── inference/
│   ├── __init__.py             ✓ (Implementado)
│   ├── model_server.py         🔜 (Futuro)
│   ├── batch_processor.py      🔜 (Futuro)
│   └── optimizers.py           🔜 (Futuro)
├── adaptation/
│   ├── __init__.py             ✓ (Implementado)
│   ├── continuous_learning.py  🔜 (Futuro)
│   ├── feedback_processor.py   🔜 (Futuro)
│   └── knowledge_distillation.py 🔜 (Futuro)
└── utils/
    ├── __init__.py             ✓ (Implementado)
    ├── model_metrics.py        🔜 (Futuro)
    └── config_templates.py     🔜 (Futuro)
```

## Fase 1: Implementação Inicial (Concluída ✓)

- **ModelHub**: Sistema de gerenciamento de modelos
  - ✅ Registro e catalogação de modelos
  - ✅ Carregamento e descarregamento de modelos
  - ✅ Gerenciamento de versões

- **Estrutura básica de diretórios e pacotes**
  - ✅ Organização dos módulos
  - ✅ Arquivos __init__.py

- **DatasetBuilder**: Preparação de datasets para fine-tuning
  - ✅ Extração de dados de pipelines
  - ✅ Formatação para diferentes tipos de modelos
  - ✅ Divisão em conjuntos de treino/validação/teste

## Fase 2: Sistema de Fine-tuning ⏳

- **TrainingManager**
  - Inicialização e configuração
  - Suporte a LoRA e QLoRA para fine-tuning eficiente
  - Tracking de métricas durante treinamento
  - Checkpoints e retomada de treinamento
  - Exportação de modelos treinados

- **ModelEvaluator**
  - Avaliação de modelos com métricas padrão
  - Comparação entre versões de modelos
  - Relatórios de desempenho

## Fase 3: Sistema de Inferência 🔜

- **ModelServer**
  - Exposição de modelos via API REST
  - Throttling e controle de acesso
  - Logging e rastreamento de requisições

- **BatchProcessor**
  - Agrupamento de requisições similares
  - Processamento paralelo
  - Priorização de requisições

- **Optimizers**
  - Quantização de modelos
  - Pruning e compressão
  - Exportação para ONNX/TensorRT

## Fase 4: Sistema de Adaptação Contínua 🔜

- **ContinuousLearner**
  - Atualização incremental de modelos
  - Aprendizado a partir de interações
  - Rotação automática de datasets

- **FeedbackProcessor**
  - Coleta de feedback de usuários
  - Processamento para ajustes de modelos
  - Detecção de lacunas de conhecimento

- **KnowledgeDistiller**
  - Transferência de conhecimento entre modelos
  - Compressão de modelos grandes para pequenos
  - Fine-tuning com supervisão de modelo

## Integração com PipelineBuilder

A integração com o sistema PipelineBuilder existente ocorrerá em várias etapas:

1. **Conexão com Pipeline → Dataset**: ⏳
   - Extração de dados de pipelines para treinamento

2. **Dataset → Modelo**: 🔜
   - Fine-tuning de modelos com dados preparados

3. **Modelo → Serviço**: 🔜
   - Exposição de modelos para consumo via API

4. **Feedback → Adaptação**: 🔜
   - Coleta e incorporação de feedback para melhoria contínua

## Próximos Passos Imediatos

1. Implementar `TrainingManager` para fine-tuning
   - Suporte inicial para modelos HuggingFace
   - Integração com ModelHub
   - Configurações para PEFT (LoRA, QLoRA)

2. Adicionar testes para modelo hub e dataset builder
   - Testes unitários para funções principais
   - Testes de integração para fluxo completo

3. Criar exemplos didáticos
   - Notebook de demonstração do fluxo completo
   - Exemplos de fine-tuning com diferentes tipos de modelos

## Requisitos e Dependências

- **Essenciais**:
  - PyTorch 2.0+
  - transformers 4.30+
  - datasets
  - accelerate
  - psutil (já utilizado pelo MemoryManager)

- **Opcionais**:
  - bitsandbytes (quantização)
  - peft (Parameter-Efficient Fine-Tuning)
  - onnxruntime (otimização ONNX)
  - triton (otimização com Triton)
  - gradio (interface de usuário)

## Considerações de Performance

- Utilizar o MemoryManager existente para otimizar uso de recursos
- Implementar carregamento sob demanda de componentes de modelos
- Configurar offloading CPU/GPU quando necessário
- Aproveitar a quantização para modelos grandes

## Documentação Contínua

À medida que o desenvolvimento ocorre, a documentação será atualizada:

- README.md principal com visão geral ✓
- Documentação específica para cada módulo
- Exemplos de uso
- Diagramas de arquitetura
 