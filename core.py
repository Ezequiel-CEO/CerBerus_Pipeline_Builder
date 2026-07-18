#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo central do PipelineBuilder.

Coordena o processo de coleta, análise e processamento de conteúdo
para criação de pipelines de treinamento.
"""

import os
import uuid
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
import traceback
from collections import defaultdict
import asyncio
import concurrent.futures
import math
from pathlib import Path

from cerberus_api.pipeline_builder.models import (
    DataSource,
    CollectionTask,
    ContentChunk,
    AnalysisResult,
    Pipeline,
    PipelineApproval,
    SourceType,
    ContentType,
    TaskStatus,
    PipelineStatus,
    ContentCategory,
)

from cerberus_api.pipeline_builder.collectors.web_collector import WebCollector
from cerberus_api.pipeline_builder.collectors.pdf_collector import PDFCollector
from cerberus_api.pipeline_builder.collectors.github_collector import GitHubCollector
from cerberus_api.pipeline_builder.collectors.youtube_collector import YouTubeCollector
from cerberus_api.pipeline_builder.analyzers.content_analyzer import ContentAnalyzer
from cerberus_api.pipeline_builder.analyzers.enhanced_analyzer import EnhancedAnalyzer
from cerberus_api.pipeline_builder.processors.pipeline_processor import (
    PipelineProcessor,
)
from cerberus_api.pipeline_builder.utils.storage import (
    ensure_dir_exists,
    save_content_chunk,
    load_content_chunk,
    save_analysis_result,
    create_pipeline_manifest,
    save_pipeline,
    load_pipeline,
    load_analysis_result,
)
from cerberus_api.utils.logging_config import APILogger

# Configurar logger
logger = APILogger("pipeline_builder")

# Definir média de tokens por caractere para diferentes tipos de conteúdo
TOKEN_RATIOS = {
    ContentType.TEXT: 0.25,  # ~4 caracteres por token para texto
    ContentType.CODE: 0.3,  # ~3.3 caracteres por token para código
    ContentType.MARKDOWN: 0.28,  # ~3.6 caracteres por token para markdown
    ContentType.HTML: 0.2,  # ~5 caracteres por token para html
    "default": 0.25,  # Padrão para outros tipos
}


def estimate_tokens(
    chunks_with_analysis: List[Tuple[ContentChunk, AnalysisResult]],
) -> int:
    """
    Estima o número de tokens em um conjunto de chunks.

    Args:
        chunks_with_analysis: Lista de tuplas (chunk, análise).

    Returns:
        Número estimado de tokens.
    """
    total_tokens = 0

    for chunk, _ in chunks_with_analysis:
        # Obter o ratio para o tipo de conteúdo
        content_type = chunk.content_type
        ratio = TOKEN_RATIOS.get(content_type, TOKEN_RATIOS["default"])

        # Calcular tokens baseado no comprimento do texto
        char_count = len(chunk.text)
        chunk_tokens = math.ceil(char_count * ratio)

        total_tokens += chunk_tokens

    return total_tokens


class PipelineBuilder:
    """
    Classe principal para construção de pipelines de treinamento.
    """

    def __init__(
        self,
        base_dir: str = "data",
        max_workers: int = 4,
        min_relevance_score: float = 0.5,
        max_pipeline_size_mb: int = 1000,
    ):
        """
        Inicializa o PipelineBuilder.

        Args:
            base_dir: Diretório base para armazenamento de dados.
            max_workers: Número máximo de workers para processamento paralelo.
            min_relevance_score: Pontuação mínima de relevância para incluir conteúdo.
            max_pipeline_size_mb: Tamanho máximo do pipeline em MB.
        """
        # Diretórios de trabalho
        self.base_dir = base_dir
        self.sources_dir = os.path.join(base_dir, "sources")
        self.content_dir = os.path.join(base_dir, "content")
        self.analysis_dir = os.path.join(base_dir, "analysis")
        self.pipelines_dir = os.path.join(base_dir, "pipelines")

        # Parâmetros de configuração
        self.max_workers = max_workers
        self.min_relevance_score = min_relevance_score
        self.max_pipeline_size_mb = max_pipeline_size_mb

        # Componentes
        self.web_collector = WebCollector(storage_dir=self.content_dir)
        self.content_analyzer = ContentAnalyzer(min_relevance_score=min_relevance_score)
        self.pipeline_processor = PipelineProcessor(
            output_dir=self.pipelines_dir,
            min_relevance_score=min_relevance_score,
            max_pipeline_size_mb=max_pipeline_size_mb,
        )

        # Criar diretórios de trabalho
        self._setup_directories()

        logger.info(f"PipelineBuilder inicializado. Base: {base_dir}")

    def _setup_directories(self) -> None:
        """
        Cria os diretórios de trabalho necessários.
        """
        ensure_dir_exists(self.base_dir)
        ensure_dir_exists(self.sources_dir)
        ensure_dir_exists(self.content_dir)
        ensure_dir_exists(self.analysis_dir)
        ensure_dir_exists(self.pipelines_dir)

    async def build_pipeline_async(
        self,
        sources: List[DataSource],
        name: str,
        description: str = "",
        categories: Optional[List[ContentCategory]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        """
        Constrói um pipeline de treinamento de forma assíncrona.

        Args:
            sources: Lista de fontes de dados para coletar.
            name: Nome do pipeline.
            description: Descrição do pipeline.
            categories: Categorias do pipeline (opcional).
            metadata: Metadados adicionais (opcional).

        Returns:
            Pipeline de treinamento criado.
        """
        logger.info(f"Iniciando construção de pipeline: {name}")

        # Etapa 1: Coleta de dados
        collection_tasks = await self._collect_content_async(sources)

        # Etapa 2: Análise de conteúdo
        all_chunks, analysis_results = await self._analyze_content_async(
            collection_tasks
        )

        # Etapa 3: Processamento e organização do pipeline
        pipeline = self.pipeline_processor.process_analyzed_content(
            chunks=all_chunks,
            analysis_results=analysis_results,
            name=name,
            description=description,
            categories=categories,
            metadata=metadata,
        )

        logger.info(f"Pipeline construído com sucesso: {name} (ID: {pipeline.id})")
        return pipeline

    def build_pipeline(
        self,
        sources: List[DataSource],
        name: str,
        description: str = "",
        categories: Optional[List[ContentCategory]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        """
        Constrói um pipeline de treinamento de forma síncrona.

        Args:
            sources: Lista de fontes de dados para coletar.
            name: Nome do pipeline.
            description: Descrição do pipeline.
            categories: Categorias do pipeline (opcional).
            metadata: Metadados adicionais (opcional).

        Returns:
            Pipeline de treinamento criado.
        """
        # Usar o asyncio para executar a versão assíncrona
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            pipeline = loop.run_until_complete(
                self.build_pipeline_async(
                    sources=sources,
                    name=name,
                    description=description,
                    categories=categories,
                    metadata=metadata,
                )
            )
            return pipeline
        finally:
            loop.close()

    async def _collect_content_async(
        self, sources: List[DataSource]
    ) -> List[CollectionTask]:
        """
        Coleta conteúdo das fontes de forma assíncrona.

        Args:
            sources: Lista de fontes de dados.

        Returns:
            Lista de tarefas de coleta.
        """
        logger.info(f"Iniciando coleta de {len(sources)} fontes")

        tasks = []
        for source in sources:
            # Criar tarefa de coleta
            task_id = str(uuid.uuid4())
            task = CollectionTask(
                id=task_id,
                source_id=source.id,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
            )
            tasks.append(task)

        # Processar fontes em paralelo
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Criar tarefas para processamento paralelo
            futures = []
            for i, source in enumerate(sources):
                task = tasks[i]
                futures.append(
                    loop.run_in_executor(
                        executor, self._collect_from_source, source, task
                    )
                )

            # Aguardar conclusão de todas as tarefas
            await asyncio.gather(*futures)

        logger.info(f"Coleta concluída para {len(sources)} fontes")
        return tasks

    def _collect_from_source(self, source: DataSource, task: CollectionTask) -> None:
        """
        Coleta conteúdo de uma fonte específica.

        Args:
            source: Fonte de dados.
            task: Tarefa de coleta.
        """
        try:
            logger.info(
                f"Coletando da fonte: {source.name} (ID: {source.id}, Tipo: {source.source_type})"
            )

            # Selecionar o coletor apropriado com base no tipo da fonte
            if source.source_type == SourceType.WEB:
                # Usar WebCollector para páginas web
                chunks = self.web_collector.collect(task, source)
            elif source.source_type == SourceType.GITHUB:
                # TODO: Implementar GitHub collector
                logger.warning(f"Coletor GitHub não implementado ainda")
                task.status = TaskStatus.FAILED
                task.error_message = "Coletor GitHub não implementado"
                return
            elif source.source_type == SourceType.PDF:
                # TODO: Implementar PDF collector
                logger.warning(f"Coletor PDF não implementado ainda")
                task.status = TaskStatus.FAILED
                task.error_message = "Coletor PDF não implementado"
                return
            elif source.source_type == SourceType.YOUTUBE:
                # TODO: Implementar YouTube collector
                logger.warning(f"Coletor YouTube não implementado ainda")
                task.status = TaskStatus.FAILED
                task.error_message = "Coletor YouTube não implementado"
                return
            else:
                # Tipo de fonte não suportado
                logger.error(f"Tipo de fonte não suportado: {source.source_type}")
                task.status = TaskStatus.FAILED
                task.error_message = (
                    f"Tipo de fonte não suportado: {source.source_type}"
                )
                return

            logger.info(
                f"Coleta concluída para fonte {source.id}. Coletados {len(chunks)} chunks"
            )

        except Exception as e:
            logger.error(f"Erro durante coleta da fonte {source.id}: {e}")
            logger.error(traceback.format_exc())
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()

    async def _analyze_content_async(
        self, collection_tasks: List[CollectionTask]
    ) -> Tuple[List[ContentChunk], List[AnalysisResult]]:
        """
        Analisa o conteúdo coletado de forma assíncrona.

        Args:
            collection_tasks: Tarefas de coleta concluídas.

        Returns:
            Tupla com listas de chunks e resultados de análise.
        """
        logger.info(
            f"Iniciando análise de conteúdo para {len(collection_tasks)} tarefas"
        )

        all_chunks = []
        all_analysis_results = []

        # Filtrar apenas tarefas concluídas com sucesso
        completed_tasks = [
            task for task in collection_tasks if task.status == TaskStatus.COMPLETED
        ]
        logger.info(
            f"Analisando {len(completed_tasks)} de {len(collection_tasks)} tarefas (tarefas concluídas com sucesso)"
        )

        if not completed_tasks:
            logger.warning("Nenhuma tarefa de coleta foi concluída com sucesso")
            return [], []

        # Processar cada tarefa concluída
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Criar tarefas para processamento paralelo
            futures = []
            for task in completed_tasks:
                futures.append(
                    loop.run_in_executor(executor, self._analyze_task_content, task)
                )

            # Aguardar conclusão de todas as tarefas e coletar resultados
            results = await asyncio.gather(*futures)

            # Combinar resultados
            for chunks, analysis_results in results:
                all_chunks.extend(chunks)
                all_analysis_results.extend(analysis_results)

        logger.info(
            f"Análise concluída. {len(all_chunks)} chunks, {len(all_analysis_results)} resultados de análise"
        )
        return all_chunks, all_analysis_results

    def _analyze_task_content(
        self, task: CollectionTask
    ) -> Tuple[List[ContentChunk], List[AnalysisResult]]:
        """
        Analisa o conteúdo coletado em uma tarefa específica.

        Args:
            task: Tarefa de coleta.

        Returns:
            Tupla com listas de chunks e resultados de análise.
        """
        chunks = []
        analysis_results = []

        try:
            logger.info(f"Analisando conteúdo da tarefa: {task.id}")

            # Carregar chunks
            if not task.content_path or not os.path.exists(task.content_path):
                logger.warning(
                    f"Caminho de conteúdo inválido para tarefa {task.id}: {task.content_path}"
                )
                return [], []

            # Iterar pelos arquivos no diretório
            for filename in os.listdir(task.content_path):
                if filename.startswith("chunk_") and filename.endswith(".json"):
                    chunk_path = os.path.join(task.content_path, filename)

                    # Carregar o chunk
                    chunk = load_content_chunk(chunk_path)
                    chunks.append(chunk)

                    # Analisar o conteúdo
                    analysis_result = self.content_analyzer.analyze(chunk)
                    analysis_results.append(analysis_result)

                    # Salvar o resultado da análise
                    analysis_dir = os.path.join(self.analysis_dir, task.id)
                    ensure_dir_exists(analysis_dir)
                    analysis_path = os.path.join(
                        analysis_dir, f"analysis_{chunk.id}.json"
                    )
                    save_analysis_result(analysis_result, analysis_path)

            logger.info(
                f"Análise concluída para tarefa {task.id}. {len(chunks)} chunks analisados"
            )
            return chunks, analysis_results

        except Exception as e:
            logger.error(f"Erro durante análise da tarefa {task.id}: {e}")
            logger.error(traceback.format_exc())
            return chunks, analysis_results

    def approve_pipeline(
        self,
        pipeline_id: str,
        user_id: str,
        approved: bool = True,
        comment: Optional[str] = None,
    ) -> PipelineApproval:
        """
        Aprova ou rejeita um pipeline.

        Args:
            pipeline_id: ID do pipeline.
            user_id: ID do usuário que está aprovando/rejeitando.
            approved: True para aprovar, False para rejeitar.
            comment: Comentário opcional.

        Returns:
            Objeto de aprovação/rejeição.
        """
        logger.info(
            f"{'Aprovando' if approved else 'Rejeitando'} pipeline: {pipeline_id}"
        )

        # Verificar se o pipeline existe
        pipeline_dir = os.path.join(self.pipelines_dir, f"pipeline_{pipeline_id}")
        metadata_path = os.path.join(pipeline_dir, "pipeline_metadata.json")

        if not os.path.exists(metadata_path):
            logger.error(f"Pipeline não encontrado: {pipeline_id}")
            raise ValueError(f"Pipeline não encontrado: {pipeline_id}")

        # Carregar metadados do pipeline
        with open(metadata_path, "r", encoding="utf-8") as f:
            pipeline_data = json.load(f)

        # Criar objeto de aprovação
        approval_id = str(uuid.uuid4())
        approval = PipelineApproval(
            id=approval_id,
            pipeline_id=pipeline_id,
            approved=approved,
            user_id=user_id,
            comment=comment,
            created_at=datetime.now(),
        )

        # Atualizar status do pipeline
        new_status = PipelineStatus.APPROVED if approved else PipelineStatus.REJECTED
        pipeline_data["status"] = new_status.value
        pipeline_data["updated_at"] = datetime.now().isoformat()

        # Adicionar aprovação aos metadados
        if "approvals" not in pipeline_data:
            pipeline_data["approvals"] = []

        pipeline_data["approvals"].append(
            {
                "id": approval.id,
                "approved": approval.approved,
                "user_id": approval.user_id,
                "comment": approval.comment,
                "created_at": approval.created_at.isoformat(),
            }
        )

        # Salvar metadados atualizados
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_data, f, ensure_ascii=False, indent=2)

        # Atualizar manifesto
        manifest_path = os.path.join(pipeline_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            manifest["status"] = new_status.value
            manifest["updated_at"] = pipeline_data["updated_at"]

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Pipeline {pipeline_id} {'aprovado' if approved else 'rejeitado'} por {user_id}"
        )
        return approval

    def list_pipelines(
        self, status: Optional[PipelineStatus] = None
    ) -> List[Dict[str, Any]]:
        """
        Lista os pipelines disponíveis.

        Args:
            status: Filtrar por status (opcional).

        Returns:
            Lista de metadados dos pipelines.
        """
        pipelines = []

        # Percorrer o diretório de pipelines
        for dirname in os.listdir(self.pipelines_dir):
            if dirname.startswith("pipeline_"):
                pipeline_dir = os.path.join(self.pipelines_dir, dirname)
                manifest_path = os.path.join(pipeline_dir, "manifest.json")

                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest = json.load(f)

                        # Filtrar por status, se especificado
                        if status is None or manifest.get("status") == status.value:
                            pipelines.append(manifest)
                    except Exception as e:
                        logger.error(f"Erro ao ler manifesto {manifest_path}: {e}")

        # Ordenar por data de criação (mais recente primeiro)
        pipelines.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return pipelines

    def get_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Obtém os metadados de um pipeline específico.

        Args:
            pipeline_id: ID do pipeline.

        Returns:
            Metadados do pipeline.
        """
        pipeline_dir = os.path.join(self.pipelines_dir, f"pipeline_{pipeline_id}")
        metadata_path = os.path.join(pipeline_dir, "pipeline_metadata.json")

        if not os.path.exists(metadata_path):
            logger.error(f"Pipeline não encontrado: {pipeline_id}")
            raise ValueError(f"Pipeline não encontrado: {pipeline_id}")

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            return metadata
        except Exception as e:
            logger.error(f"Erro ao ler metadados do pipeline {pipeline_id}: {e}")
            raise

def load_model(model_path: str, **kwargs) -> Dict[str, Any]:
    """
    Carrega um modelo do pipeline.

    Args:
        model_path: Caminho para o modelo
        **kwargs: Argumentos adicionais

    Returns:
        Dict com informações do modelo carregado
    """
    try:
        if not os.path.exists(model_path):
            return {
                "status": "error",
                "message": f"Modelo não encontrado: {model_path}",
                "model_path": model_path,
            }

        # Simula carregamento do modelo
        return {
            "status": "success",
            "message": f"Modelo carregado: {model_path}",
            "model_path": model_path,
            "loaded_at": datetime.now().isoformat(),
            "kwargs": kwargs,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erro ao carregar modelo: {e}",
            "model_path": model_path,
        }

def save_model(model_data: Any, model_path: str, **kwargs) -> Dict[str, Any]:
    """
    Salva um modelo do pipeline.

    Args:
        model_data: Dados do modelo a serem salvos
        model_path: Caminho onde salvar o modelo
        **kwargs: Argumentos adicionais

    Returns:
        Dict com informações do modelo salvo
    """
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Simula salvamento do modelo
        with open(model_path, "w") as f:
            json.dump(
                {
                    "model_data": str(model_data)[:1000],  # Limita o tamanho
                    "saved_at": datetime.now().isoformat(),
                    "kwargs": kwargs,
                },
                f,
                indent=2,
            )

        return {
            "status": "success",
            "message": f"Modelo salvo: {model_path}",
            "model_path": model_path,
            "saved_at": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erro ao salvar modelo: {e}",
            "model_path": model_path,
        }


def create_pipeline(
    name: str, config: Dict[str, Any], output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a new processing pipeline with the given configuration.

    Args:
        name: Name of the pipeline
        config: Pipeline configuration dictionary
        output_dir: Optional output directory path

    Returns:
        Dictionary with pipeline details
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "pipelines", name)

    os.makedirs(output_dir, exist_ok=True)

    pipeline = {
        "name": name,
        "config": config,
        "output_dir": output_dir,
        "status": "created",
    }

    # Save pipeline config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(pipeline, f, indent=2)

    return pipeline


def run_pipeline(pipeline: Dict[str, Any], input_data: Any = None) -> Dict[str, Any]:
    """
    Runs a processing pipeline with optional input data.

    Args:
        pipeline: Pipeline configuration dictionary
        input_data: Optional input data for the pipeline

    Returns:
        Dictionary with pipeline execution results
    """
    logger.info(f"Running pipeline: {pipeline['name']}")

    try:
        # Load pipeline config
        config = pipeline["config"]
        output_dir = pipeline["output_dir"]

        # Initialize results
        results = {
            "pipeline_name": pipeline["name"],
            "status": "running",
            "output_dir": output_dir,
            "results": {},
        }

        # Execute pipeline steps
        for step in config.get("steps", []):
            step_name = step["name"]
            step_type = step["type"]
            step_config = step.get("config", {})

            logger.info(f"Executing step: {step_name} ({step_type})")

            # Execute step based on type
            if step_type == "inference":
                results["results"][step_name] = _run_inference_step(
                    step_config, input_data
                )
            elif step_type == "processing":
                results["results"][step_name] = _run_processing_step(
                    step_config, input_data
                )
            else:
                logger.warning(f"Unknown step type: {step_type}")

        results["status"] = "completed"

        # Save results
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return {"pipeline_name": pipeline["name"], "status": "failed", "error": str(e)}


def _run_inference_step(config: Dict[str, Any], input_data: Any) -> Dict[str, Any]:
    """Executes an inference step"""
    return {
        "type": "inference",
        "status": "simulated",
        "input": str(input_data)[:100] if input_data else None,
    }


def _run_processing_step(config: Dict[str, Any], input_data: Any) -> Dict[str, Any]:
    """Executes a processing step"""
    return {
        "type": "processing",
        "status": "simulated",
        "input": str(input_data)[:100] if input_data else None,
    }
