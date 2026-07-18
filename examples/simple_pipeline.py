#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemplo simples de uso do PipelineBuilder.

Este script demonstra como utilizar o PipelineBuilder para criar um pipeline
de treinamento a partir de fontes web.
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


def criar_pipeline_simples(base_dir="data"):
    """
    Cria um pipeline de treinamento simples a partir de fontes web.

    Args:
        base_dir: Diretório base para armazenamento de dados.
    """
    print("Iniciando criação de pipeline simples...")

    # Criar o builder
    builder = PipelineBuilder(
        base_dir=base_dir,
        max_workers=4,
        min_relevance_score=0.4,
        max_pipeline_size_mb=50,
    )

    # Definir fontes de dados
    sources = [
        DataSource(
            id=str(uuid.uuid4()),
            name="Documentação Python",
            source_type=SourceType.WEB,
            location="https://docs.python.org/pt-br/3/tutorial/",
            description="Documentação oficial do Python em português",
            metadata={"follow_links": True, "max_pages": 5, "depth": 1},
            created_at=datetime.now(),
        ),
        DataSource(
            id=str(uuid.uuid4()),
            name="Blog sobre IA",
            source_type=SourceType.WEB,
            location="https://machinelearningmastery.com/blog/",
            description="Blog sobre Machine Learning e IA",
            metadata={"follow_links": True, "max_pages": 3, "depth": 1},
            created_at=datetime.now(),
        ),
    ]

    # Criar pipeline
    try:
        pipeline = builder.build_pipeline(
            sources=sources,
            name="Tutorial Python + IA",
            description="Pipeline combinando tutoriais de Python e conceitos de IA",
            categories=[ContentCategory.TUTORIALS, ContentCategory.PROGRAMMING],
            metadata={
                "creator": "PipelineBuilder Example",
                "purpose": "Demonstração",
                "target_model": "cerberus-test-1.0",
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

        if not pipeline.sources:
            print("Aviso: Nenhuma fonte adicionada ao pipeline.")
            return None

        # Aprovar o pipeline
        print("\nAprovando pipeline...")
        approval = builder.approve_pipeline(
            pipeline_id=pipeline.id,
            user_id="usuario-exemplo",
            approved=True,
            comment="Pipeline aprovado para uso em treinamento",
        )

        print(f"Pipeline aprovado: {approval.approved}")

        # Listar pipelines
        print("\nListando todos os pipelines:")
        pipelines = builder.list_pipelines()
        for p in pipelines:
            print(f"- {p['name']} (ID: {p['pipeline_id']}, Status: {p['status']})")

        return pipeline.id

    except Exception as e:
        print(f"Erro durante criação do pipeline: {e}")
        import traceback

        traceback.print_exc()
        return None


def explorar_pipeline(pipeline_id, base_dir="data"):
    """
    Explora o conteúdo de um pipeline criado.

    Args:
        pipeline_id: ID do pipeline a explorar.
        base_dir: Diretório base para armazenamento de dados.
    """
    if not pipeline_id:
        print("Nenhum ID de pipeline fornecido para exploração")
        return

    print(f"\nExplorando conteúdo do pipeline {pipeline_id}...")

    builder = PipelineBuilder(base_dir=base_dir)

    try:
        # Obter metadados
        metadata = builder.get_pipeline(pipeline_id)
        print(f"Pipeline: {metadata['name']}")
        print(f"Descrição: {metadata['description']}")
        print(f"Status: {metadata['status']}")

        # Explorar diretório de dados
        data_path = metadata["data_path"]
        if os.path.exists(data_path):
            print(f"\nArquivos no diretório de dados:")
            for root, dirs, files in os.walk(data_path):
                level = root.replace(data_path, "").count(os.sep)
                indent = " " * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                sub_indent = " " * 4 * (level + 1)
                for f in files:
                    print(f"{sub_indent}{f}")

        # Explorar um pouco do conteúdo
        processed_dir = os.path.join(data_path, "processed")
        if os.path.exists(processed_dir):
            dataset_path = os.path.join(processed_dir, "dataset.jsonl")
            if os.path.exists(dataset_path):
                print("\nAmostra do conteúdo processado (primeiras 3 linhas):")
                with open(dataset_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= 3:
                            break
                        print(f"Linha {i+1}: {line[:100]}...")

    except Exception as e:
        print(f"Erro ao explorar pipeline: {e}")


if __name__ == "__main__":
    # Registrar início
    start_time = time.time()

    # Diretório base
    base_dir = "data"

    # Criar pipeline
    pipeline_id = criar_pipeline_simples(base_dir=base_dir)

    # Explorar conteúdo
    if pipeline_id:
        explorar_pipeline(pipeline_id, base_dir=base_dir)

    # Registrar tempo total
    execution_time = time.time() - start_time
    print(f"\nTempo total de execução: {execution_time:.2f} segundos")
