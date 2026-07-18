#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface de linha de comando para o PipelineBuilder.

Este módulo fornece comandos para interagir com o PipelineBuilder
através do terminal, facilitando o uso do sistema.
"""

import os
import sys
import json
import uuid
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from cerberus_api.pipeline_builder.core import PipelineBuilder
from cerberus_api.pipeline_builder.models import (
    DataSource,
    SourceType,
    ContentCategory,
    PipelineStatus,
)
from cerberus_api.pipeline_builder.ui import PipelineDashboard
from cerberus_api.utils.logging_config import APILogger

# Configurar logger
logger = APILogger("pipeline_cli")


def parse_arguments():
    """
    Analisa os argumentos da linha de comando.

    Returns:
        Namespace com os argumentos parseados.
    """
    parser = argparse.ArgumentParser(description="PipelineBuilder CLI")
    subparsers = parser.add_subparsers(dest="command", help="Comando para executar")

    # Comando para criar um pipeline
    create_parser = subparsers.add_parser("create", help="Criar um novo pipeline")
    create_parser.add_argument("--name", required=True, help="Nome do pipeline")
    create_parser.add_argument(
        "--description", required=True, help="Descrição do pipeline"
    )
    create_parser.add_argument(
        "--categories", required=True, help="Categorias, separadas por vírgula"
    )
    create_parser.add_argument(
        "--sources", required=True, help="Arquivo JSON com as fontes de dados"
    )
    create_parser.add_argument(
        "--base-dir", default="data", help="Diretório base para armazenamento"
    )
    create_parser.add_argument(
        "--max-workers", type=int, default=4, help="Número máximo de workers"
    )
    create_parser.add_argument(
        "--min-relevance",
        type=float,
        default=0.5,
        help="Pontuação mínima de relevância",
    )

    # Comando para listar pipelines
    list_parser = subparsers.add_parser("list", help="Listar pipelines existentes")
    list_parser.add_argument(
        "--base-dir", default="data", help="Diretório base para armazenamento"
    )
    list_parser.add_argument("--status", help="Filtrar por status")
    list_parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Formato de saída"
    )

    # Comando para obter detalhes de um pipeline
    get_parser = subparsers.add_parser("get", help="Obter detalhes de um pipeline")
    get_parser.add_argument("--id", required=True, help="ID do pipeline")
    get_parser.add_argument(
        "--base-dir", default="data", help="Diretório base para armazenamento"
    )
    get_parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Formato de saída"
    )

    # Comando para aprovar um pipeline
    approve_parser = subparsers.add_parser(
        "approve", help="Aprovar ou rejeitar um pipeline"
    )
    approve_parser.add_argument("--id", required=True, help="ID do pipeline")
    approve_parser.add_argument("--user", required=True, help="ID do usuário aprovador")
    approve_parser.add_argument(
        "--approve", type=bool, default=True, help="Aprovar (True) ou rejeitar (False)"
    )
    approve_parser.add_argument("--comment", default="", help="Comentário opcional")
    approve_parser.add_argument(
        "--base-dir", default="data", help="Diretório base para armazenamento"
    )

    # Comando para iniciar a interface web
    ui_parser = subparsers.add_parser("dashboard", help="Iniciar interface gráfica")
    ui_parser.add_argument(
        "--base-dir", default="data", help="Diretório base para armazenamento"
    )
    ui_parser.add_argument(
        "--port", type=int, default=7860, help="Porta para a interface web"
    )

    # Comando para executar exemplo simples
    example_parser = subparsers.add_parser("example", help="Executar um exemplo")
    example_parser.add_argument(
        "--type", choices=["simple", "multi"], default="simple", help="Tipo de exemplo"
    )
    example_parser.add_argument(
        "--base-dir", default="data", help="Diretório base para armazenamento"
    )

    return parser.parse_args()


def create_pipeline(args):
    """
    Cria um novo pipeline com base nos argumentos fornecidos.

    Args:
        args: Argumentos da linha de comando.
    """
    try:
        # Inicializar o builder
        builder = PipelineBuilder(
            base_dir=args.base_dir,
            max_workers=args.max_workers,
            min_relevance_score=args.min_relevance,
        )

        # Carregar fontes do arquivo JSON
        with open(args.sources, "r") as f:
            sources_data = json.load(f)

        # Converter para objetos DataSource
        sources = []
        for source_data in sources_data:
            source_type = getattr(SourceType, source_data.get("source_type", "WEB"))
            source = DataSource(
                id=source_data.get("id", str(uuid.uuid4())),
                name=source_data["name"],
                source_type=source_type,
                location=source_data["location"],
                description=source_data.get("description", ""),
                metadata=source_data.get("metadata", {}),
                created_at=datetime.now(),
            )
            sources.append(source)

        # Converter categorias
        categories = []
        for category_name in args.categories.split(","):
            category_name = category_name.strip().upper()
            if hasattr(ContentCategory, category_name):
                categories.append(getattr(ContentCategory, category_name))

        # Criar pipeline
        print(f"Criando pipeline '{args.name}' com {len(sources)} fontes...")

        pipeline = builder.build_pipeline(
            sources=sources,
            name=args.name,
            description=args.description,
            categories=categories,
        )

        print(f"\nPipeline criado com sucesso!")
        print(f"ID: {pipeline.id}")
        print(f"Nome: {pipeline.name}")
        print(f"Status: {pipeline.status}")
        print(f"Categorias: {[c.value for c in pipeline.categories]}")
        print(f"Tamanho: {pipeline.size_bytes / (1024*1024):.2f} MB")
        print(f"Tokens estimados: {pipeline.estimated_tokens}")
        print(f"Diretório de dados: {pipeline.data_path}")

    except Exception as e:
        logger.error(f"Erro ao criar pipeline: {e}")
        print(f"Erro: {e}")
        sys.exit(1)


def list_pipelines(args):
    """
    Lista pipelines existentes.

    Args:
        args: Argumentos da linha de comando.
    """
    try:
        # Inicializar o builder
        builder = PipelineBuilder(base_dir=args.base_dir)

        # Listar pipelines
        pipelines = builder.list_pipelines()

        # Filtrar por status, se especificado
        if args.status:
            status = getattr(PipelineStatus, args.status.upper(), None)
            if status:
                pipelines = [p for p in pipelines if p["status"] == status.value]

        # Formatar saída
        if args.format == "json":
            print(json.dumps(pipelines, indent=2))
        else:
            # Formato tabela
            print(f"\nPipelines ({len(pipelines)}):")
            print("-" * 100)
            print(
                f"{'ID':<36} | {'Nome':<30} | {'Status':<12} | {'Tamanho':<10} | {'Criado em':<20}"
            )
            print("-" * 100)

            for p in pipelines:
                pipeline_id = p["id"]
                name = p["name"][:30]
                status = p["status"]
                size = f"{p.get('size', 0) / (1024*1024):.2f} MB"
                created = p.get("created_at", "")[:19]

                print(
                    f"{pipeline_id:<36} | {name:<30} | {status:<12} | {size:<10} | {created:<20}"
                )

            print("-" * 100)

    except Exception as e:
        logger.error(f"Erro ao listar pipelines: {e}")
        print(f"Erro: {e}")
        sys.exit(1)


def get_pipeline(args):
    """
    Obtém detalhes de um pipeline.

    Args:
        args: Argumentos da linha de comando.
    """
    try:
        # Inicializar o builder
        builder = PipelineBuilder(base_dir=args.base_dir)

        # Obter detalhes do pipeline
        pipeline = builder.get_pipeline(args.id)

        # Formatar saída
        if args.format == "json":
            print(json.dumps(pipeline, indent=2))
        else:
            # Formato tabela
            print(f"\nDetalhes do Pipeline {pipeline['id']}:")
            print("-" * 80)

            print(f"Nome: {pipeline['name']}")
            print(f"Descrição: {pipeline['description']}")
            print(f"Status: {pipeline['status']}")
            print(f"Categorias: {pipeline.get('categories', [])}")
            print(f"Tamanho: {pipeline.get('size', 0) / (1024*1024):.2f} MB")
            print(f"Tokens estimados: {pipeline.get('estimated_tokens', 0)}")
            print(f"Criado em: {pipeline.get('created_at', '')}")
            print(f"Diretório de dados: {pipeline.get('data_path', '')}")

            if pipeline.get("approval"):
                print("\nStatus de Aprovação:")
                print(f"Aprovado: {pipeline['approval'].get('approved', False)}")
                print(f"Aprovado por: {pipeline['approval'].get('user_id', 'N/A')}")
                print(
                    f"Data de aprovação: {pipeline['approval'].get('approved_at', 'N/A')}"
                )
                print(f"Comentário: {pipeline['approval'].get('comment', '')}")

            print("\nFontes de Dados:")
            for source_id in pipeline.get("sources", []):
                print(f"- {source_id}")

            print("-" * 80)

    except Exception as e:
        logger.error(f"Erro ao obter detalhes do pipeline: {e}")
        print(f"Erro: {e}")
        sys.exit(1)


def approve_pipeline(args):
    """
    Aprova ou rejeita um pipeline.

    Args:
        args: Argumentos da linha de comando.
    """
    try:
        # Inicializar o builder
        builder = PipelineBuilder(base_dir=args.base_dir)

        # Aprovar/rejeitar pipeline
        action = "Aprovando" if args.approve else "Rejeitando"
        print(f"{action} pipeline {args.id}...")

        approval = builder.approve_pipeline(
            pipeline_id=args.id,
            user_id=args.user,
            approved=args.approve,
            comment=args.comment,
        )

        status = "aprovado" if approval.approved else "rejeitado"
        print(f"Pipeline {status} com sucesso!")
        print(f"Comentário: {approval.comment}")

    except Exception as e:
        logger.error(f"Erro ao aprovar/rejeitar pipeline: {e}")
        print(f"Erro: {e}")
        sys.exit(1)


def start_dashboard(args):
    """
    Inicia a interface gráfica.

    Args:
        args: Argumentos da linha de comando.
    """
    try:
        print(f"Iniciando dashboard na porta {args.port}...")

        # Criar e iniciar o dashboard
        dashboard = PipelineDashboard(base_dir=args.base_dir, port=args.port)
        dashboard.run()

    except Exception as e:
        logger.error(f"Erro ao iniciar dashboard: {e}")
        print(f"Erro: {e}")
        sys.exit(1)


def run_example(args):
    """
    Executa um exemplo pré-configurado.

    Args:
        args: Argumentos da linha de comando.
    """
    try:
        if args.type == "simple":
            print("Executando exemplo simples...")
            from cerberus_api.pipeline_builder.examples.simple_pipeline import (
                criar_pipeline_simples,
            )

            pipeline_id = criar_pipeline_simples(base_dir=args.base_dir)

            if pipeline_id:
                print(f"Exemplo executado com sucesso! Pipeline ID: {pipeline_id}")
            else:
                print("Falha ao executar exemplo simples.")

        elif args.type == "multi":
            print("Executando exemplo multifonte...")
            from cerberus_api.pipeline_builder.examples.multi_source_pipeline import (
                criar_pipeline_multifonte,
            )

            pipeline_id = criar_pipeline_multifonte()

            if pipeline_id:
                print(
                    f"Exemplo multifonte executado com sucesso! Pipeline ID: {pipeline_id}"
                )

                # Perguntar se deseja iniciar o dashboard
                response = input(
                    "\nDeseja iniciar o dashboard para visualizar o pipeline? (s/n): "
                )
                if response.lower() == "s":
                    start_dashboard(
                        argparse.Namespace(base_dir=args.base_dir, port=7860)
                    )
            else:
                print("Falha ao executar exemplo multifonte.")

    except Exception as e:
        logger.error(f"Erro ao executar exemplo: {e}")
        print(f"Erro: {e}")
        sys.exit(1)


def main():
    """
    Função principal da CLI.
    """
    args = parse_arguments()

    # Verificar o comando e chamar a função apropriada
    if args.command == "create":
        create_pipeline(args)
    elif args.command == "list":
        list_pipelines(args)
    elif args.command == "get":
        get_pipeline(args)
    elif args.command == "approve":
        approve_pipeline(args)
    elif args.command == "dashboard":
        start_dashboard(args)
    elif args.command == "example":
        run_example(args)
    else:
        print("Comando não reconhecido. Use --help para ver os comandos disponíveis.")
        sys.exit(1)


if __name__ == "__main__":
    main()
