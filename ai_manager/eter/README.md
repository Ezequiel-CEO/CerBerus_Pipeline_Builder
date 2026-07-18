# Èter - Agente Inteligente do CerBerus FMK

## Visão Geral

Èter é o componente central de inteligência artificial do framework CerBerus FMK, projetado para funcionar como um "piloto automático" ou assistente inteligente que pode controlar e interagir com todos os componentes do sistema.

O nome "Èter" vem do conceito histórico do "éter luminífero", uma substância que se acreditava preencher todo o espaço e servir como meio para a propagação da luz. De forma similar, o agente Èter permeia todo o framework CerBerus FMK, facilitando a comunicação e operação entre seus diferentes componentes.

## Funcionalidades Principais

- **Processamento de Linguagem Natural**: Compreensão e execução de comandos em linguagem natural
- **Integração com o Framework**: Acesso a todos os componentes do CerBerus FMK
- **Memória e Contexto**: Armazenamento de conversas, conhecimento e contexto de execução
- **Aprendizado**: Capacidade de melhorar com base em interações e feedback
- **Execução de Pipelines**: Criação e gerenciamento de pipelines de processamento
- **Segurança**: Mecanismos para garantir operações seguras

## Arquitetura

O módulo Èter é organizado em vários subcomponentes:

```
eter/
├── agent.py              # Classe principal EterAgent
├── core_integration.py   # Ponte de integração com o framework
├── context/              # Gerenciamento de contexto e memória
├── interface/            # Interfaces de usuário (CLI, API, etc.)
├── learning/             # Componentes de aprendizado
├── llm/                  # Integração com modelos de linguagem
├── security/             # Componentes de segurança
└── tools/                # Ferramentas e utilitários
```

### Integração com o Framework

Um dos principais diferenciais do Èter é sua capacidade de integração profunda com o framework CerBerus FMK. Através da classe `EterFrameworkBridge`, o agente tem acesso a todos os componentes do framework, incluindo:

- **PipelineBuilder**: Para criação e execução de pipelines
- **ModelHub**: Para gerenciamento de modelos de IA
- **MemoryManager**: Para otimização de recursos computacionais
- **Analisadores**: Para processamento de conteúdo
- **Coletores**: Para obtenção de dados de diversas fontes

## Utilização

### Inicialização Básica

```python
from cerberus_api.pipeline_builder.ai_manager.eter import initialize

# Inicializar o agente com configurações padrão
eter = initialize(user_id="usuario1")

# Processar um comando em linguagem natural
resultado = eter.process_user_command("Coletar conteúdo do site https://exemplo.com")
print(resultado)
```

### Configuração Avançada

```python
from cerberus_api.pipeline_builder.ai_manager.eter import initialize

# Inicializar com configurações personalizadas
eter = initialize(
    user_id="usuario_avancado",
    model_dir="/caminho/para/modelos",
    knowledge_dir="/caminho/para/conhecimento",
    workspace_dir="/caminho/para/workspace",
    framework_integration=True,
    log_level="DEBUG"
)

# Acessar componentes do framework diretamente
pipeline_builder = eter.get_framework_component("cerberus_api.pipeline_builder.core.PipelineBuilder")
```

## Processamento de Comandos

O Èter pode processar comandos em linguagem natural e convertê-los em ações concretas no framework. Por exemplo:

- "Coletar conteúdo do site X"
- "Analisar este texto e extrair entidades"
- "Criar um pipeline para processar os dados da pasta Y"
- "Executar o script de análise Z com os parâmetros A e B"

## Integração com Modelos de Linguagem

O Èter pode utilizar diferentes tipos de modelos de linguagem:

1. **Modelos GGUF**: Para processamento local de linguagem natural
2. **APIs de LLM**: Para processamento em nuvem (quando configurado)
3. **Modelos Específicos**: Para tarefas como classificação, resumo, etc.

## Segurança

O Èter implementa várias camadas de segurança:

- **Validação de Comandos**: Verifica comandos perigosos antes da execução
- **Isolamento**: Executa em ambiente controlado
- **Registros**: Mantém logs detalhados de todas as operações
- **Permissões**: Respeita níveis de acesso configurados

## Desenvolvimento

Para estender o Èter com novos recursos:

1. **Novas Ferramentas**: Adicione no diretório `tools/`
2. **Novas Integrações**: Estenda a classe `EterFrameworkBridge`
3. **Novos Modelos**: Adicione suporte no diretório `llm/`

## Status do Projeto

O Èter está em desenvolvimento ativo e novas funcionalidades são adicionadas regularmente. A versão atual é 0.1.0.

---

**© CerBerus FMK - Todos os direitos reservados** 