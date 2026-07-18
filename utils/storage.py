#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilitários de armazenamento para o PipelineBuilder.

Responsável por salvar, recuperar e gerenciar conteúdo coletado.
"""

import os
import json
import shutil
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

from cerberus_api.pipeline_builder.models import ContentChunk, Pipeline, AnalysisResult
from cerberus_api.utils.logging_config import APILogger

# Configurar logger
logger = APILogger("storage_utils")


def ensure_dir_exists(directory: str) -> None:
    """
    Garante que um diretório existe, criando-o se necessário.

    Args:
        directory: Caminho do diretório.
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Diretório criado: {directory}")


def save_content_chunk(chunk: ContentChunk, filepath: str) -> None:
    """
    Salva um chunk de conteúdo em formato JSON.

    Args:
        chunk: Objeto ContentChunk a ser salvo.
        filepath: Caminho do arquivo para salvar.
    """
    # Garantir que o diretório existe
    ensure_dir_exists(os.path.dirname(filepath))

    try:
        # Converter para dict
        chunk_dict = chunk.dict()

        # Converter datetime para string
        for key, value in chunk_dict.items():
            if isinstance(value, datetime):
                chunk_dict[key] = value.isoformat()

        # Salvar como JSON
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chunk_dict, f, ensure_ascii=False, indent=2)

        logger.debug(f"Chunk salvo em: {filepath}")

    except Exception as e:
        logger.error(f"Erro ao salvar chunk: {e}")
        raise


def load_content_chunk(filepath: str) -> ContentChunk:
    """
    Carrega um chunk de conteúdo de um arquivo JSON.

    Args:
        filepath: Caminho do arquivo para carregar.

    Returns:
        Objeto ContentChunk carregado.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            chunk_dict = json.load(f)

        # Converter string para datetime
        for key in ["created_at"]:
            if key in chunk_dict and isinstance(chunk_dict[key], str):
                chunk_dict[key] = datetime.fromisoformat(chunk_dict[key])

        # Criar objeto ContentChunk
        return ContentChunk(**chunk_dict)

    except Exception as e:
        logger.error(f"Erro ao carregar chunk de {filepath}: {e}")
        raise


def save_analysis_result(result: AnalysisResult, filepath: str) -> None:
    """
    Salva um resultado de análise em formato JSON.

    Args:
        result: Objeto AnalysisResult a ser salvo.
        filepath: Caminho do arquivo para salvar.
    """
    ensure_dir_exists(os.path.dirname(filepath))

    try:
        # Converter para dict
        result_dict = result.dict()

        # Converter datetime para string
        for key, value in result_dict.items():
            if isinstance(value, datetime):
                result_dict[key] = value.isoformat()

        # Salvar como JSON
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

        logger.debug(f"Resultado de análise salvo em: {filepath}")

    except Exception as e:
        logger.error(f"Erro ao salvar resultado de análise: {e}")
        raise


def load_analysis_result(filepath: str) -> AnalysisResult:
    """
    Carrega um resultado de análise de um arquivo JSON.

    Args:
        filepath: Caminho do arquivo para carregar.

    Returns:
        Objeto AnalysisResult carregado.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            result_dict = json.load(f)

        # Converter string para datetime
        for key in ["created_at"]:
            if key in result_dict and isinstance(result_dict[key], str):
                result_dict[key] = datetime.fromisoformat(result_dict[key])

        # Criar objeto AnalysisResult
        return AnalysisResult(**result_dict)

    except Exception as e:
        logger.error(f"Erro ao carregar resultado de análise de {filepath}: {e}")
        raise


def save_pipeline_data(pipeline: Pipeline, output_dir: str) -> str:
    """
    Salva os dados completos de um pipeline incluindo metadados.

    Args:
        pipeline: Objeto Pipeline a ser salvo.
        output_dir: Diretório base para salvar o pipeline.

    Returns:
        Caminho do diretório onde o pipeline foi salvo.
    """
    # Criar diretório específico para o pipeline
    pipeline_dir = os.path.join(output_dir, f"pipeline_{pipeline.id}")
    ensure_dir_exists(pipeline_dir)

    try:
        # Salvar metadados do pipeline
        pipeline_dict = pipeline.dict()

        # Converter datetime para string
        for key, value in pipeline_dict.items():
            if isinstance(value, datetime):
                pipeline_dict[key] = value.isoformat()

        # Salvar metadados
        with open(
            os.path.join(pipeline_dir, "pipeline_metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(pipeline_dict, f, ensure_ascii=False, indent=2)

        # Criar diretório para os dados do pipeline
        pipeline_data_dir = os.path.join(pipeline_dir, "data")
        ensure_dir_exists(pipeline_data_dir)

        # Criar diretório para configurações
        pipeline_config_dir = os.path.join(pipeline_dir, "config")
        ensure_dir_exists(pipeline_config_dir)

        # Atualizar caminhos no objeto pipeline
        pipeline.data_path = pipeline_data_dir
        pipeline.config_path = pipeline_config_dir

        logger.info(f"Pipeline salvo em: {pipeline_dir}")
        return pipeline_dir

    except Exception as e:
        logger.error(f"Erro ao salvar pipeline: {e}")
        raise


def merge_chunks_to_dataset(
    chunks_dir: str, output_file: str, format: str = "jsonl"
) -> int:
    """
    Combina múltiplos chunks em um único arquivo de dataset.

    Args:
        chunks_dir: Diretório contendo os chunks.
        output_file: Caminho do arquivo de saída.
        format: Formato do dataset ("jsonl", "text" ou "csv").

    Returns:
        Número de chunks mesclados.
    """
    ensure_dir_exists(os.path.dirname(output_file))

    count = 0

    try:
        if format == "jsonl":
            with open(output_file, "w", encoding="utf-8") as out_f:
                for filename in os.listdir(chunks_dir):
                    if filename.endswith(".json"):
                        filepath = os.path.join(chunks_dir, filename)
                        chunk = load_content_chunk(filepath)

                        # Criar registro para o dataset
                        record = {
                            "id": chunk.id,
                            "text": chunk.text,
                            "metadata": {
                                "source": chunk.source_location,
                                "content_type": chunk.content_type.value,
                                "created_at": chunk.created_at.isoformat(),
                                **chunk.metadata,
                            },
                        }

                        # Escrever como JSONL
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1

        elif format == "text":
            with open(output_file, "w", encoding="utf-8") as out_f:
                for filename in os.listdir(chunks_dir):
                    if filename.endswith(".json"):
                        filepath = os.path.join(chunks_dir, filename)
                        chunk = load_content_chunk(filepath)

                        # Adicionar separador
                        out_f.write(f"--- Chunk ID: {chunk.id} ---\n")
                        out_f.write(f"Source: {chunk.source_location}\n")
                        out_f.write(f"Type: {chunk.content_type.value}\n\n")

                        # Adicionar conteúdo
                        out_f.write(chunk.text)
                        out_f.write("\n\n")
                        count += 1

        elif format == "csv":
            import csv

            with open(output_file, "w", encoding="utf-8", newline="") as out_f:
                writer = csv.writer(out_f)
                # Escrever cabeçalho
                writer.writerow(["id", "text", "source", "content_type", "created_at"])

                for filename in os.listdir(chunks_dir):
                    if filename.endswith(".json"):
                        filepath = os.path.join(chunks_dir, filename)
                        chunk = load_content_chunk(filepath)

                        # Escrever como CSV
                        writer.writerow(
                            [
                                chunk.id,
                                chunk.text,
                                chunk.source_location,
                                chunk.content_type.value,
                                chunk.created_at.isoformat(),
                            ]
                        )
                        count += 1

        else:
            raise ValueError(
                f"Formato desconhecido: {format}. Use 'jsonl', 'text' ou 'csv'."
            )

        logger.info(
            f"Dataset criado em {output_file} com {count} chunks no formato {format}"
        )
        return count

    except Exception as e:
        logger.error(f"Erro ao mesclar chunks para dataset: {e}")
        raise


def create_pipeline_manifest(pipeline: Pipeline, output_file: str) -> None:
    """
    Cria um arquivo de manifesto para o pipeline.

    Args:
        pipeline: Objeto Pipeline.
        output_file: Caminho do arquivo de manifesto.
    """
    ensure_dir_exists(os.path.dirname(output_file))

    try:
        # Criar o arquivo de manifesto
        manifest = {
            "pipeline_id": pipeline.id,
            "name": pipeline.name,
            "description": pipeline.description,
            "status": pipeline.status.value,
            "created_at": pipeline.created_at.isoformat(),
            "updated_at": pipeline.updated_at.isoformat(),
            "categories": [c.value for c in pipeline.categories],
            "size_bytes": pipeline.size_bytes,
            "estimated_tokens": pipeline.estimated_tokens,
            "source_count": len(pipeline.sources),
            "sources": pipeline.sources,  # Já é uma lista de IDs (strings)
        }

        # Salvar manifesto
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        logger.info(f"Manifesto do pipeline criado em: {output_file}")

    except Exception as e:
        logger.error(f"Erro ao criar manifesto do pipeline: {e}")
        raise


def save_pipeline(pipeline: "Pipeline", base_dir: str = "data") -> None:
    """
    Salva os metadados de um pipeline.

    Args:
        pipeline: Pipeline a ser salvo.
        base_dir: Diretório base para armazenamento.
    """
    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline.id)
    os.makedirs(pipeline_dir, exist_ok=True)

    # Converter para dicionário serializável
    pipeline_dict = pipeline.model_dump()

    # Converter categorias para valores string
    pipeline_dict["categories"] = [c.value for c in pipeline.categories]
    pipeline_dict["status"] = pipeline.status.value

    # Se houver aprovação, converter para dicionário serializável
    if pipeline.approval:
        pipeline_dict["approval"] = pipeline.approval.model_dump()

    # Salvar metadados do pipeline
    with open(os.path.join(pipeline_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(pipeline_dict, f, ensure_ascii=False, indent=2, default=str)


def load_pipeline(pipeline_id: str, base_dir: str = "data") -> Dict:
    """
    Carrega os metadados de um pipeline.

    Args:
        pipeline_id: ID do pipeline a ser carregado.
        base_dir: Diretório base para armazenamento.

    Returns:
        Dicionário com os metadados do pipeline.
    """
    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_id)

    if not os.path.exists(pipeline_dir):
        raise FileNotFoundError(f"Pipeline {pipeline_id} não encontrado")

    metadata_path = os.path.join(pipeline_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadados do pipeline {pipeline_id} não encontrados")

    with open(metadata_path, "r", encoding="utf-8") as f:
        pipeline_data = json.load(f)

    return pipeline_data
