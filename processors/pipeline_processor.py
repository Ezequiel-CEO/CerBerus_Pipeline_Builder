#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Processador de pipelines para o PipelineBuilder.

Organiza o conteúdo analisado em pipelines de treinamento.
"""

import os
import uuid
import json
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
import logging
from collections import defaultdict
import shutil

from cerberus_api.pipeline_builder.models import (
    Pipeline,
    ContentChunk,
    AnalysisResult,
    PipelineStatus,
    ContentCategory,
    TaskStatus,
)
from cerberus_api.pipeline_builder.utils.storage import (
    ensure_dir_exists,
    save_pipeline_data,
    merge_chunks_to_dataset,
    create_pipeline_manifest,
)
from cerberus_api.utils.logging_config import APILogger
from cerberus_api.pipeline_builder.utils.nlp_utils import estimate_tokens

# Configurar logger
logger = APILogger("pipeline_processor")


class PipelineProcessor:
    """
    Processador para organizar conteúdo em pipelines de treinamento.
    """

    def __init__(
        self,
        output_dir: str = "data/pipelines",
        min_relevance_score: float = 0.5,
        min_pipeline_size_mb: int = 5,
        max_pipeline_size_mb: int = 1000,
        target_token_count: int = 1_000_000,
    ):
        """
        Inicializa o processador de pipelines.

        Args:
            output_dir: Diretório base para salvar pipelines.
            min_relevance_score: Pontuação mínima de relevância para incluir conteúdo.
            min_pipeline_size_mb: Tamanho mínimo do pipeline em MB.
            max_pipeline_size_mb: Tamanho máximo do pipeline em MB.
            target_token_count: Número alvo de tokens para um pipeline.
        """
        self.output_dir = output_dir
        self.min_relevance_score = min_relevance_score
        self.min_pipeline_size = (
            min_pipeline_size_mb * 1024 * 1024
        )  # Converter para bytes
        self.max_pipeline_size = (
            max_pipeline_size_mb * 1024 * 1024
        )  # Converter para bytes
        self.target_token_count = target_token_count

        ensure_dir_exists(output_dir)
        logger.info(f"PipelineProcessor inicializado. Saída: {output_dir}")

    def process_analyzed_content(
        self,
        chunks: List[ContentChunk],
        analysis_results: List[AnalysisResult],
        name: str,
        description: str = "",
        categories: Optional[List[ContentCategory]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        """
        Processa o conteúdo analisado e cria um pipeline de treinamento.

        Args:
            chunks: Lista de chunks de conteúdo.
            analysis_results: Lista de resultados de análise correspondentes.
            name: Nome do pipeline.
            description: Descrição do pipeline.
            categories: Categorias do pipeline (opcional).
            metadata: Metadados adicionais (opcional).

        Returns:
            Pipeline criado.
        """
        logger.info(f"Processando pipeline: {name}")

        # Criar mapeamento de chunk_id para resultado de análise
        analysis_map = {result.chunk_id: result for result in analysis_results}

        # Filtrar chunks por relevância
        relevant_chunks = []
        for chunk in chunks:
            if chunk.id in analysis_map:
                result = analysis_map[chunk.id]
                if result.relevance_score >= self.min_relevance_score:
                    relevant_chunks.append((chunk, result))

        logger.info(
            f"Encontrados {len(relevant_chunks)} chunks relevantes de {len(chunks)} total"
        )

        if not relevant_chunks:
            logger.warning(f"Nenhum chunk relevante encontrado para o pipeline {name}")
            return self._create_empty_pipeline(name, description, categories, metadata)

        # Calcular estatísticas do pipeline
        total_size = sum(chunk.length for chunk, _ in relevant_chunks)
        estimated_tokens = estimate_tokens(relevant_chunks)

        # Verificar tamanho do pipeline
        if total_size < self.min_pipeline_size:
            logger.warning(
                f"Pipeline muito pequeno: {total_size / (1024*1024):.2f} MB (mínimo: {self.min_pipeline_size / (1024*1024)} MB)"
            )

        if total_size > self.max_pipeline_size:
            logger.warning(
                f"Pipeline muito grande: {total_size / (1024*1024):.2f} MB (máximo: {self.max_pipeline_size / (1024*1024)} MB)"
            )
            # Ordenar por relevância e cortar
            relevant_chunks.sort(key=lambda x: x[1].relevance_score, reverse=True)

            # Cortar até atingir o tamanho máximo
            cumulative_size = 0
            cutoff_index = 0

            for i, (chunk, _) in enumerate(relevant_chunks):
                cumulative_size += chunk.length
                if cumulative_size > self.max_pipeline_size:
                    cutoff_index = i
                    break

            # Manter apenas até o índice de corte
            if cutoff_index > 0:
                relevant_chunks = relevant_chunks[:cutoff_index]
                logger.info(f"Pipeline truncado para {len(relevant_chunks)} chunks")

        # Identificar fontes dos dados
        sources = set()
        for chunk, _ in relevant_chunks:
            if chunk.source_location:
                sources.add(chunk.source_location)

        # Determinar categorias do pipeline se não foram especificadas
        if not categories:
            # Contar ocorrências de categorias nos chunks relevantes
            category_counts = defaultdict(int)
            for _, result in relevant_chunks:
                for category in result.categories:
                    category_counts[category] += 1

            # Selecionar as categorias mais frequentes (no máximo 3)
            pipeline_categories = [
                category
                for category, _ in sorted(
                    category_counts.items(), key=lambda x: x[1], reverse=True
                )[:3]
            ]

            # Se não houver categorias identificadas, usar OTHER
            if not pipeline_categories:
                pipeline_categories = [ContentCategory.OTHER]
        else:
            pipeline_categories = categories

        # Criar o objeto Pipeline
        pipeline_id = str(uuid.uuid4())
        pipeline = Pipeline(
            id=pipeline_id,
            name=name,
            description=description,
            status=PipelineStatus.DRAFT,
            sources=list(sources),
            data_path="",  # Será definido durante o salvamento
            config_path="",  # Será definido durante o salvamento
            size_bytes=total_size,
            estimated_tokens=estimated_tokens,
            categories=pipeline_categories,
            metadata=metadata or {},
            created_at=datetime.now(),
        )

        # Finalizar o pipeline
        pipeline.description = description
        pipeline.categories = list(set(pipeline_categories))
        pipeline.estimated_tokens = estimated_tokens
        pipeline.status = PipelineStatus.READY
        pipeline.updated_at = datetime.now()

        # Salvar os dados
        pipeline_dir = save_pipeline_data(pipeline, self.output_dir)

        # Organizar conteúdo se houver chunks relevantes
        if relevant_chunks:
            self._organize_pipeline_content(pipeline, relevant_chunks, pipeline_dir)
            logger.info(
                f"Pipeline criado: {pipeline.name} (ID: {pipeline.id}, Tamanho: {pipeline.size_bytes / (1024*1024):.2f} MB)"
            )
            return pipeline
        else:
            logger.warning("Nenhum conteúdo relevante encontrado após filtragem.")
            return self._create_empty_pipeline(name, description, categories, metadata)

    def _organize_pipeline_content(
        self,
        pipeline: Pipeline,
        relevant_chunks: List[Tuple[ContentChunk, AnalysisResult]],
        pipeline_dir: str,
    ) -> None:
        """
        Organiza o conteúdo do pipeline em diferentes formatos e estruturas.

        Args:
            pipeline: Objeto Pipeline.
            relevant_chunks: Lista de tuplas (chunk, análise) relevantes.
            pipeline_dir: Diretório do pipeline.
        """
        # Criar diretório para dados brutos
        raw_dir = os.path.join(pipeline.data_path, "raw")
        ensure_dir_exists(raw_dir)

        # Criar diretório para dados processados
        processed_dir = os.path.join(pipeline.data_path, "processed")
        ensure_dir_exists(processed_dir)

        # Separar chunks por categoria
        chunks_by_category = defaultdict(list)
        for chunk, result in relevant_chunks:
            for category in result.categories:
                chunks_by_category[category].append((chunk, result))

        # Salvar chunks brutos por categoria
        for category, category_chunks in chunks_by_category.items():
            category_dir = os.path.join(raw_dir, category.value.lower())
            ensure_dir_exists(category_dir)

            # Salvar cada chunk como um arquivo JSON separado
            for i, (chunk, result) in enumerate(category_chunks):
                chunk_file = os.path.join(category_dir, f"chunk_{i}_{chunk.id}.json")
                with open(chunk_file, "w", encoding="utf-8") as f:
                    # Combinar chunk e resultado da análise
                    combined = {"chunk": chunk.dict(), "analysis": result.dict()}

                    # Converter datetimes para string
                    for section in ["chunk", "analysis"]:
                        for key, value in combined[section].items():
                            if isinstance(value, datetime):
                                combined[section][key] = value.isoformat()

                    json.dump(combined, f, ensure_ascii=False, indent=2)

        # Criar datasets em diferentes formatos

        # 1. Dataset completo em formato JSONL
        jsonl_path = os.path.join(processed_dir, "dataset.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for chunk, result in relevant_chunks:
                entry = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "source": chunk.source_location,
                    "content_type": chunk.content_type.value,
                    "relevance": result.relevance_score,
                    "categories": [c.value for c in result.categories],
                    "topics": result.topics,
                    "summary": result.summary,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 2. Dataset de texto plano
        text_path = os.path.join(processed_dir, "dataset.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            for chunk, _ in relevant_chunks:
                f.write(f"--- Fonte: {chunk.source_location} ---\n\n")
                f.write(chunk.text)
                f.write("\n\n")

        # 3. Datasets por categoria
        for category, category_chunks in chunks_by_category.items():
            category_path = os.path.join(
                processed_dir, f"dataset_{category.value.lower()}.jsonl"
            )
            with open(category_path, "w", encoding="utf-8") as f:
                for chunk, result in category_chunks:
                    entry = {
                        "id": chunk.id,
                        "text": chunk.text,
                        "source": chunk.source_location,
                        "relevance": result.relevance_score,
                        "topics": result.topics,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 4. Criar arquivo de configuração para o pipeline
        config = {
            "pipeline_id": pipeline.id,
            "name": pipeline.name,
            "description": pipeline.description,
            "created_at": pipeline.created_at.isoformat(),
            "size_bytes": pipeline.size_bytes,
            "estimated_tokens": pipeline.estimated_tokens,
            "categories": [c.value for c in pipeline.categories],
            "dataset_files": {
                "jsonl": os.path.relpath(jsonl_path, pipeline.config_path),
                "text": os.path.relpath(text_path, pipeline.config_path),
                "categories": {
                    category.value: os.path.relpath(
                        os.path.join(
                            processed_dir, f"dataset_{category.value.lower()}.jsonl"
                        ),
                        pipeline.config_path,
                    )
                    for category in chunks_by_category.keys()
                },
            },
            "statistics": {
                "total_chunks": len(relevant_chunks),
                "chunks_by_category": {
                    category.value: len(category_chunks)
                    for category, category_chunks in chunks_by_category.items()
                },
                "avg_relevance": sum(
                    result.relevance_score for _, result in relevant_chunks
                )
                / len(relevant_chunks),
                "size_mb": pipeline.size_bytes / (1024 * 1024),
            },
            "metadata": pipeline.metadata,
        }

        config_path = os.path.join(pipeline.config_path, "pipeline_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # 5. Criar manifesto do pipeline
        manifest_path = os.path.join(pipeline_dir, "manifest.json")
        create_pipeline_manifest(pipeline, manifest_path)

        # Atualizar status do pipeline
        pipeline.status = PipelineStatus.READY
        pipeline.updated_at = datetime.now()

        # Salvar metadados atualizados
        with open(
            os.path.join(pipeline_dir, "pipeline_metadata.json"), "w", encoding="utf-8"
        ) as f:
            pipeline_dict = pipeline.dict()

            # Converter datetime para string
            for key, value in pipeline_dict.items():
                if isinstance(value, datetime):
                    pipeline_dict[key] = value.isoformat()

            json.dump(pipeline_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"Pipeline organizado e salvo em: {pipeline_dir}")

    def _create_empty_pipeline(
        self,
        name: str,
        description: str,
        categories: Optional[List[ContentCategory]],
        metadata: Optional[Dict[str, Any]],
    ) -> Pipeline:
        """
        Cria um pipeline vazio quando não há conteúdo relevante.

        Args:
            name: Nome do pipeline.
            description: Descrição do pipeline.
            categories: Categorias do pipeline.
            metadata: Metadados adicionais.

        Returns:
            Pipeline vazio.
        """
        pipeline_id = str(uuid.uuid4())
        pipeline = Pipeline(
            id=pipeline_id,
            name=name,
            description=description,
            status=PipelineStatus.DRAFT,
            sources=[],
            data_path="",
            config_path="",
            size_bytes=0,
            estimated_tokens=0,
            categories=categories or [ContentCategory.OTHER],
            metadata=metadata or {"error": "Sem conteúdo relevante suficiente"},
            created_at=datetime.now(),
        )

        # Salvar o pipeline vazio
        pipeline_dir = save_pipeline_data(pipeline, self.output_dir)

        # Criar diretórios vazios
        ensure_dir_exists(os.path.join(pipeline.data_path, "raw"))
        ensure_dir_exists(os.path.join(pipeline.data_path, "processed"))

        # Criar manifesto
        manifest_path = os.path.join(pipeline_dir, "manifest.json")
        create_pipeline_manifest(pipeline, manifest_path)

        # Criar arquivo de configuração vazio
        config = {
            "pipeline_id": pipeline.id,
            "name": pipeline.name,
            "description": pipeline.description,
            "created_at": pipeline.created_at.isoformat(),
            "error": "Sem conteúdo relevante suficiente",
            "size_bytes": 0,
            "estimated_tokens": 0,
            "categories": [c.value for c in pipeline.categories],
            "statistics": {
                "total_chunks": 0,
                "chunks_by_category": {},
                "avg_relevance": 0,
                "size_mb": 0,
            },
            "metadata": pipeline.metadata,
        }

        config_path = os.path.join(pipeline.config_path, "pipeline_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        logger.warning(f"Pipeline vazio criado: {pipeline.name} (ID: {pipeline.id})")
        return pipeline
