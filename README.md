# PipelineBuilder v1.1

O PipelineBuilder é um módulo para o CerBerusFMK que automatiza a coleta, análise, processamento e organização de conteúdos externos em pipelines de aprendizado para fine-tuning.

## Funcionalidades

O PipelineBuilder oferece as seguintes funcionalidades:

- **Coleta Inteligente**: Coleta conteúdo relevante de diversas fontes externas, como páginas web, PDFs, repositórios GitHub e vídeos do YouTube.

- **Análise de Relevância**: Avalia automaticamente a relevância e qualidade do conteúdo coletado para fins de treinamento.

- **Organização Estruturada**: Organiza o conteúdo em pipelines de treinamento prontos para uso.

- **Controle de Qualidade**: Implementa verificações e filtros para garantir a qualidade dos dados de treinamento.

- **Visualização Avançada**: Dashboard interativo para explorar e analisar o conteúdo dos pipelines através de gráficos e visualizações.

- **Análise de Similaridade**: Visualização de relações entre chunks através de grafos de similaridade.

## Estrutura do Módulo

O PipelineBuilder é organizado da seguinte forma:

```
cerberus_api/pipeline_builder/
├── __init__.py              # Inicialização do módulo
├── core.py                  # Implementação principal do PipelineBuilder
├── models.py                # Modelos de dados usando Pydantic
├── collectors/              # Coletores para diferentes tipos de fontes
│   ├── web_collector.py     # Coletor para conteúdo web
│   ├── pdf_collector.py     # Coletor para documentos PDF
│   ├── github_collector.py  # Coletor para repositórios GitHub
│   └── youtube_collector.py # Coletor para vídeos do YouTube
├── analyzers/               # Analisadores de conteúdo
│   ├── content_analyzer.py  # Analisador básico
│   └── enhanced_analyzer.py # Analisador avançado com embeddings
├── processors/              # Processadores para organizar pipelines
│   └── pipeline_processor.py # Organizador de pipelines
├── utils/                   # Utilitários
│   ├── storage.py           # Funções para armazenamento
│   └── nlp_utils.py         # Utilitários para processamento de linguagem natural
├── ui/                      # Interface de usuário
│   └── pipeline_dashboard.py # Dashboard com Gradio
├── cli.py                   # Interface de linha de comando
├── examples/                # Exemplos de uso
│   ├── simple_pipeline.py   # Exemplo simples de criação de pipeline
│   └── multi_source_pipeline.py # Exemplo com múltiplas fontes
├── tests/                   # Testes automatizados
│   ├── test_collectors.py   # Testes para coletores
│   ├── test_analyzers.py    # Testes para analisadores
│   └── test_processors.py   # Testes para processadores
├── CHANGELOG.md             # Registro de alterações
└── requirements.txt         # Dependências do módulo
```

## Instalação

1. Instale as dependências necessárias:

```bash
pip install -r cerberus_api/pipeline_builder/requirements.txt
```

2. Baixe os modelos de linguagem do spaCy:

```bash
python -m spacy download pt_core_news_sm
python -m spacy download en_core_web_sm
```

3. Para utilizar todas as funcionalidades de visualização, instale as dependências opcionais:

```bash
pip install gradio matplotlib networkx numpy scikit-learn sentence-transformers
```

## Uso Básico

```python
from cerberus_api.pipeline_builder.core import PipelineBuilder
from cerberus_api.pipeline_builder.models import DataSource, SourceType, ContentCategory

# Criar o builder
builder = PipelineBuilder(
    base_dir="data",
    max_workers=4,
    min_relevance_score=0.5
)

# Definir fontes de dados
sources = [
    DataSource(
        id="source-1",
        name="Documentação Python",
        source_type=SourceType.WEB,
        location="https://docs.python.org/pt-br/3/tutorial/",
        description="Documentação oficial do Python em português",
        metadata={
            "follow_links": True,
            "max_pages": 5
        }
    )
]

# Criar pipeline
pipeline = builder.build_pipeline(
    sources=sources,
    name="Tutorial Python",
    description="Pipeline de treinamento baseado na documentação do Python",
    categories=[ContentCategory.TUTORIALS, ContentCategory.PROGRAMMING]
)

print(f"Pipeline criado: {pipeline.id}")
print(f"Tamanho: {pipeline.size / (1024*1024):.2f} MB")
print(f"Tokens estimados: {pipeline.estimated_tokens}")
```

## Interface de Linha de Comando (CLI)

O PipelineBuilder inclui uma interface de linha de comando para facilitar o uso sem necessidade de programação. Para utilizar a CLI:

```bash
python -m cerberus_api.pipeline_builder.cli [comando] [opções]
```

### Comandos Disponíveis:

#### Criar um novo pipeline

```bash
python -m cerberus_api.pipeline_builder.cli create --name "Nome do Pipeline" \
    --description "Descrição do pipeline" \
    --categories "TUTORIALS,PROGRAMMING" \
    --sources path/to/sources.json
```

O arquivo `sources.json` deve conter um array JSON com as fontes de dados:

```json
[
  {
    "name": "Documentação Python",
    "source_type": "WEB",
    "location": "https://docs.python.org/pt-br/3/tutorial/",
    "description": "Documentação oficial do Python em português",
    "metadata": {
      "follow_links": true,
      "max_pages": 5
    }
  }
]
```

Um exemplo completo está disponível em `examples/sources_example.json`.

#### Listar pipelines existentes

```bash
python -m cerberus_api.pipeline_builder.cli list
python -m cerberus_api.pipeline_builder.cli list --status APPROVED --format json
```

#### Obter detalhes de um pipeline

```bash
python -m cerberus_api.pipeline_builder.cli get --id [pipeline-id]
```

#### Aprovar ou rejeitar um pipeline

```bash
python -m cerberus_api.pipeline_builder.cli approve --id [pipeline-id] --user "user-id" --approve true --comment "Aprovado para treinamento"
```

#### Iniciar o dashboard

```bash
python -m cerberus_api.pipeline_builder.cli dashboard --port 7860
```

#### Executar exemplos

```bash
python -m cerberus_api.pipeline_builder.cli example --type simple
python -m cerberus_api.pipeline_builder.cli example --type multi
```

## Vercel Web Analytics

O PipelineBuilder agora suporta integração com Vercel Web Analytics para monitorar o uso do dashboard. Para habilitar:

1. Crie um arquivo `.env` na raiz do projeto (copie de `.env.example`)
2. Defina `VERCEL_ANALYTICS_ENABLED=true`
3. Deploy no Vercel ou execute localmente com Vercel CLI

Para instruções detalhadas, consulte [VERCEL_ANALYTICS_SETUP.md](./VERCEL_ANALYTICS_SETUP.md).

## Dashboard Interativo

O PipelineBuilder inclui um dashboard interativo criado com Gradio, que oferece as seguintes funcionalidades:

### Abas do Dashboard

1. **Pipelines**: Lista todos os pipelines disponíveis com detalhes como ID, nome, status, tamanho, tokens estimados e categorias.

2. **Criar Pipeline**: Interface gráfica para criar novos pipelines, permitindo definir fontes de dados em formato JSON.

3. **Processar Pipeline**: Interface para iniciar o processamento de um pipeline (coleta, análise e organização de dados).

4. **Explorar Conteúdo**: Visualiza o conteúdo de um pipeline com:
   - Informações sobre as fontes de dados
   - Amostra de conteúdo coletado
   - Grafo de similaridade entre chunks (requer bibliotecas opcionais)
   - Resultados de análise e palavras-chave

5. **Análise Avançada**: Visualizações detalhadas dos resultados de análise, incluindo:
   - Pontuações de relevância dos chunks
   - Distribuição de tamanho dos chunks
   - Visualização 2D de embeddings (usando TSNE)
   - Palavras-chave mais comuns

6. **Estatísticas**: Gráficos comparativos entre diferentes pipelines:
   - Contagem de chunks por pipeline
   - Tokens estimados por pipeline
   - Distribuição de tipos de fontes
   - Comparação de tamanho relativo

7. **Ajuda**: Documentação básica sobre o uso do dashboard.

Para iniciar o dashboard:

```bash
python -m cerberus_api.pipeline_builder.cli dashboard --port 7860
```

ou a partir do código:

```python
from cerberus_api.pipeline_builder.ui import PipelineDashboard

dashboard = PipelineDashboard(base_dir="data", port=7860)
dashboard.run()
```

## Coletores Disponíveis

- **WebCollector**: Coleta conteúdo de páginas web, blogs e artigos online.
- **PDFCollector**: Extrai conteúdo de documentos PDF locais ou remotos.
- **GitHubCollector**: Coleta código e documentação de repositórios GitHub.
- **YouTubeCollector**: Extrai metadados e transcrições de vídeos do YouTube.

## Analisadores de Relevância

O módulo inclui dois analisadores de relevância:

- **ContentAnalyzer**: Implementação básica que avalia relevância baseada em palavras-chave e estatísticas de texto.
- **EnhancedAnalyzer**: Implementação avançada com embeddings e técnicas de NLP para melhor avaliação de qualidade e relevância semântica.

Para utilizar o analisador avançado:

```python
from cerberus_api.pipeline_builder.analyzers import EnhancedAnalyzer

builder = PipelineBuilder(
    base_dir="data",
    analyzer=EnhancedAnalyzer(
        min_relevance_score=0.5,
        calculate_similarity=True,  # Calcula similaridade entre chunks
        generate_summary=True,      # Gera resumos automáticos
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2"  # Modelo multilíngue
    )
)
```

## Testes

O PipelineBuilder agora inclui testes abrangentes para todos os componentes:

```bash
# Executar todos os testes
python -m unittest discover -s cerberus_api/pipeline_builder/tests

# Executar testes específicos
python -m unittest cerberus_api/pipeline_builder/tests/test_analyzers.py
python -m unittest cerberus_api/pipeline_builder/tests/test_processors.py
python -m unittest cerberus_api/pipeline_builder/tests/test_collectors.py
```

## Exemplos Práticos

Para um exemplo completo, consulte o arquivo `examples/simple_pipeline.py`. Para um exemplo utilizando múltiplas fontes (web, PDF, GitHub, YouTube), veja `examples/multi_source_pipeline.py`.

## Visualizações

O módulo inclui visualizações avançadas dos dados coletados:

- **Grafos de similaridade**: Visualização das relações entre chunks baseada em similaridade semântica
- **Embeddings 2D**: Visualização de embeddings através de TSNE para análise de clusters
- **Estatísticas comparativas**: Gráficos comparando diferentes pipelines
- **Palavras-chave**: Visualização de frequência de palavras-chave

## Histórico de Alterações

Para detalhes sobre as alterações e versões, consulte o arquivo [CHANGELOG.md](./CHANGELOG.md). 