# Pipeline Builder - Registro de Alterações

## [0.2.0] - 2025-04-07

### Adicionado
- Testes completos para os analisadores (ContentAnalyzer e EnhancedAnalyzer)
- Testes para os processadores (PipelineProcessor)
- Nova aba de "Processar Pipeline" na interface de usuário
- Visualização avançada de resultados de análise com gráficos
- Visualização de estatísticas melhorada com múltiplos gráficos
- Visualização de grafos de similaridade entre chunks
- Interface de usuário aprimorada com tema personalizado
- Aba de ajuda com documentação básica na interface

### Melhorado
- Função de exploração de conteúdo do pipeline com visualização mais detalhada
- Estatísticas de pipeline com gráficos comparativos
- Visualização de análise com TSNE para embeddings
- Documentação em todos os componentes
- Interface geral mais intuitiva e responsiva

### Corrigido
- Problemas de renderização em alguns gráficos
- Tratamento de erros mais robusto em todo o sistema
- Manipulação de exceções durante o processamento de pipeline

## [0.1.0] - 2023-04-01

### Adicionado
- Implementação inicial do PipelineBuilder
- Coletores básicos: WebCollector, PDFCollector, GitHubCollector, YouTubeCollector
- Analisadores básicos: ContentAnalyzer, EnhancedAnalyzer
- Processador de pipeline básico
- Interface de linha de comando (CLI)
- Interface de usuário básica com Gradio
- Sistema de armazenamento persistente
- Testes básicos para coletores

### Pendente
- Testes de integração completos
- Documentação detalhada de cada módulo
- Guia de contribuição
- Expansão das opções da interface
- Suporte para mais tipos de fontes (áudio, vídeo)
- Melhor integração com modelos de linguagem
- Ferramentas avançadas de pré-processamento
- Otimizações de desempenho e memória 