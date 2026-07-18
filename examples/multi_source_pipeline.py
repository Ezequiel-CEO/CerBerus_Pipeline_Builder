#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemplo de pipeline multifonte para o PipelineBuilder.

Este script demonstra como criar um pipeline que combina
diferentes fontes de dados (web, PDF, GitHub, YouTube).
"""

import os
import sys
import uuid
import time
from datetime import datetime

# Adicionar diretório pai ao path para importação
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from cerberus_api.pipeline_builder.core import PipelineBuilder
from cerberus_api.pipeline_builder.models import DataSource, SourceType, ContentCategory


def criar_pipeline_multifonte():
    """
    Cria um pipeline de treinamento combinando diferentes fontes.
    """
    print("Iniciando criação de pipeline multifonte...")

    # Criar o builder
    builder = PipelineBuilder(
        base_dir="data",
        max_workers=4,
        min_relevance_score=0.4,
        max_pipeline_size_mb=100,  # Pipeline maior para comportar múltiplas fontes
    )

    # Definir fontes de dados
    sources = [
        # Fonte Web
        DataSource(
            id=str(uuid.uuid4()),
            name="Documentação Python",
            source_type=SourceType.WEB,
            location="https://docs.python.org/pt-br/3/tutorial/",
            description="Documentação oficial do Python em português",
            metadata={"follow_links": True, "max_pages": 3, "depth": 1},
            created_at=datetime.now(),
        ),
        # Fonte PDF (um PDF público do governo brasileiro)
        DataSource(
            id=str(uuid.uuid4()),
            name="Manual de Redação",
            source_type=SourceType.PDF,
            location="https://www.gov.br/planalto/pt-br/centrais-de-conteudo/publicacoes/manual-de-redacao-da-presidencia-da-republica/manual-de-redacao-da-presidencia_3a-edicao_2018.pdf",
            description="Manual de Redação da Presidência da República",
            metadata={},
            created_at=datetime.now(),
        ),
        # Fonte GitHub
        DataSource(
            id=str(uuid.uuid4()),
            name="Repositório Flask",
            source_type=SourceType.GITHUB,
            location="https://github.com/pallets/flask",
            description="Repositório do framework Flask",
            metadata={"max_files": 20},
            created_at=datetime.now(),
        ),
        # Fonte YouTube (vídeo sobre Python)
        DataSource(
            id=str(uuid.uuid4()),
            name="Tutorial Python",
            source_type=SourceType.YOUTUBE,
            location="https://www.youtube.com/watch?v=rfscVS0vtbw",
            description="Tutorial de Python para iniciantes",
            metadata={},
            created_at=datetime.now(),
        ),
    ]

    # Criar pipeline
    try:
        pipeline = builder.build_pipeline(
            sources=sources,
            name="Pipeline Multifontes",
            description="Pipeline combinando dados de web, PDF, GitHub e YouTube",
            categories=[
                ContentCategory.TUTORIALS,
                ContentCategory.PROGRAMMING,
                ContentCategory.DOCUMENTATION,
            ],
            metadata={
                "creator": "PipelineBuilder Example",
                "purpose": "Demonstração multifontes",
                "target_model": "cerberus-test-2.0",
            },
        )

        print(f"\nPipeline criado com sucesso!")
        print(f"ID: {pipeline.id}")
        print(f"Nome: {pipeline.name}")
        print(f"Status: {pipeline.status}")
        print(f"Categorias: {[c.value for c in pipeline.categories]}")
        print(f"Tamanho: {pipeline.size_bytes / (1024*1024):.2f} MB")
        print(f"Tokens estimados: {pipeline.estimated_tokens}")
        print(f"Diretório de dados: {pipeline.data_path}")

        # Aprovar o pipeline
        print("\nAprovando pipeline...")
        approval = builder.approve_pipeline(
            pipeline_id=pipeline.id,
            user_id="usuario-exemplo",
            approved=True,
            comment="Pipeline multifonte aprovado para uso em treinamento",
        )

        print(f"Pipeline aprovado: {approval.approved}")

        return pipeline.id

    except Exception as e:
        print(f"Erro durante criação do pipeline: {e}")
        import traceback

        traceback.print_exc()
        return None


def analisar_estatisticas_pipeline(pipeline_id):
    """
    Analisa as estatísticas do pipeline multifonte.

    Args:
        pipeline_id: ID do pipeline a analisar.
    """
    if not pipeline_id:
        print("Nenhum ID de pipeline fornecido para análise")
        return

    print(f"\nAnalisando estatísticas do pipeline {pipeline_id}...")

    builder = PipelineBuilder(base_dir="data")

    try:
        # Obter metadados
        metadata = builder.get_pipeline(pipeline_id)
        print(f"Pipeline: {metadata['name']}")

        # Analisar distribuição por fonte
        if os.path.exists(metadata["data_path"]):
            raw_dir = os.path.join(metadata["data_path"], "raw")
            if os.path.exists(raw_dir):
                print("\nDistribuição de conteúdo por tipo:")

                # Contar arquivos por categoria
                category_sizes = {}
                for root, dirs, files in os.walk(raw_dir):
                    category = os.path.basename(root)
                    if category == "raw":  # Ignorar o diretório raiz
                        continue

                    size = 0
                    file_count = 0
                    for file in files:
                        if file.endswith(".json"):
                            file_path = os.path.join(root, file)
                            size += os.path.getsize(file_path)
                            file_count += 1

                    if file_count > 0:
                        size_mb = size / (1024 * 1024)
                        category_sizes[category] = (file_count, size_mb)

                # Exibir estatísticas por categoria
                for category, (file_count, size_mb) in category_sizes.items():
                    print(f"- {category}: {file_count} arquivos, {size_mb:.2f} MB")

        # Analisar distribuição por tipo de fonte
        sources = metadata.get("sources", [])
        print(f"\nFontes utilizadas ({len(sources)}):")
        for source_id in sources:
            # Em uma implementação completa, buscaríamos detalhes de cada fonte
            print(f"- {source_id}")

    except Exception as e:
        print(f"Erro ao analisar pipeline: {e}")


def iniciar_dashboard():
    """
    Inicia o dashboard para visualizar o pipeline.
    """
    try:
        from cerberus_api.pipeline_builder.ui import PipelineDashboard

        print("\nIniciando dashboard para visualização...")

        dashboard = PipelineDashboard(base_dir="data", port=7860)
        dashboard.run()

    except ImportError:
        print(
            "\nNão foi possível iniciar o dashboard. Verifique se gradio está instalado:"
        )
        print("pip install gradio")
    except Exception as e:
        print(f"Erro ao iniciar dashboard: {e}")


if __name__ == "__main__":
    # Registrar início
    start_time = time.time()

    # Criar pipeline
    pipeline_id = criar_pipeline_multifonte()

    # Analisar estatísticas
    if pipeline_id:
        analisar_estatisticas_pipeline(pipeline_id)

        # Perguntar se deseja iniciar o dashboard
        resposta = input(
            "\nDeseja iniciar o dashboard para visualizar o pipeline? (s/n): "
        )
        if resposta.lower() == "s":
            iniciar_dashboard()

    # Registrar tempo total
    execution_time = time.time() - start_time
    print(f"\nTempo total de execução: {execution_time:.2f} segundos")
