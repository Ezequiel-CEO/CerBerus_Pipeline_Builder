# Testes do Gerenciador de Memória

Este diretório contém testes automatizados para o módulo de gerenciamento de memória e sua integração com o analisador aprimorado.

## Visão Geral dos Testes

Os testes cobrem dois componentes principais:

1. **MemoryManager** (`test_memory_manager.py`)
2. **Integração EnhancedAnalyzer com MemoryManager** (`test_enhanced_analyzer.py`)

## Testes do MemoryManager

O arquivo `test_memory_manager.py` contém testes unitários para verificar:

- **Detecção de recursos**: Teste da detecção correta de dispositivos (CPU/GPU)
- **Monitoramento de memória**: Validação dos relatórios de uso de RAM/VRAM
- **Otimização de lotes**: Verificação do cálculo adaptativo de tamanho de lotes
- **Cache de embeddings**: Testes do sistema de cache para vetores de embedding
- **Decisão de streaming**: Validação da lógica para determinar quando usar processamento em streaming
- **Instância global**: Testes da instância global do gerenciador de memória

### Executando os Testes do MemoryManager

```bash
python -m cerberus_api.pipeline_builder.tests.test_memory_manager
```

## Testes de Integração com EnhancedAnalyzer

O arquivo `test_enhanced_analyzer.py` contém testes para verificar a integração do gerenciador de memória com o analisador:

- **Logging de memória**: Verificação do registro de uso de memória durante a análise
- **Otimização de lotes**: Teste da otimização de lotes durante o processamento
- **Decisão de streaming**: Validação da decisão de usar processamento em streaming
- **Carregamento sob demanda**: Verificação do carregamento de modelos sob demanda
- **Cache de embeddings**: Teste do uso de cache para embeddings
- **Processamento eficiente**: Validação do processamento de chunks grandes

### Executando os Testes de Integração

```bash
python -m cerberus_api.pipeline_builder.tests.test_enhanced_analyzer
```

## Cobertura de Testes

Os testes cobrem as principais funcionalidades do gerenciador de memória:

- Monitoramento de recursos
- Otimização de processamento
- Carregamento e gerenciamento de modelos
- Cache de embeddings
- Estratégias de processamento (em lote vs. streaming)
- Limpeza e recuperação de memória 