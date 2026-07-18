#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DatasetBuilder: Prepara datasets para fine-tuning.

Converte chunks e análises obtidos pelo PipelineBuilder em datasets
formatados para treinamento de modelos.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import hashlib
import random

import numpy as np

from cerberus_api.pipeline_builder.models import (
    ContentChunk,
    AnalysisResult,
    Pipeline,
    ContentCategory,
    ContentType,
)

from cerberus_api.pipeline_builder.utils.storage import (
    load_content_chunk,
    load_analysis_result,
    load_pipeline,
)

from cerberus_api.utils.logging_config import APILogger

logger = APILogger("dataset_builder")


class DatasetBuilder:
    """
    Prepara datasets para fine-tuning a partir de pipelines.

    Converte conteúdo coletado e analisado em formatos adequados
    para treinamento de diferentes tipos de modelos.
    """

    def __init__(
        self,
        output_dir: str = "datasets",
        min_relevance_score: float = 0.7,
        test_split: float = 0.1,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Inicializa o DatasetBuilder.

        Args:
            output_dir: Diretório para salvar datasets
            min_relevance_score: Pontuação mínima de relevância para incluir conteúdo
            test_split: Proporção do dataset para teste
            val_split: Proporção do dataset para validação
            seed: Semente para reprodutibilidade
        """
        self.output_dir = os.path.abspath(output_dir)
        self.min_relevance_score = min_relevance_score
        self.test_split = test_split
        self.val_split = val_split
        self.seed = seed

        # Configurar diretório de saída
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            f"DatasetBuilder inicializado. Diretório de saída: {self.output_dir}"
        )

    def build_from_pipeline(
        self,
        pipeline: Union[Pipeline, str],
        dataset_name: str,
        format_type: str = "instruction",
        categories: Optional[List[ContentCategory]] = None,
        content_types: Optional[List[ContentType]] = None,
        max_length: Optional[int] = None,
        include_metadata: bool = True,
        additional_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Constrói um dataset a partir de um pipeline.

        Args:
            pipeline: Pipeline ou ID do pipeline
            dataset_name: Nome para o dataset
            format_type: Formato do dataset ("instruction", "completion", "qa", etc.)
            categories: Categorias de conteúdo a incluir (opcional)
            content_types: Tipos de conteúdo a incluir (opcional)
            max_length: Comprimento máximo de texto (opcional)
            include_metadata: Incluir metadados no dataset
            additional_filters: Filtros adicionais para chunks

        Returns:
            Informações sobre o dataset criado
        """
        # Carregar pipeline se for um ID
        if isinstance(pipeline, str):
            pipeline = load_pipeline(pipeline)
            if not pipeline:
                logger.error(f"Pipeline não encontrado: {pipeline}")
                raise ValueError(f"Pipeline não encontrado: {pipeline}")

        # Preparar diretório para o dataset
        dataset_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        logger.info(
            f"Construindo dataset '{dataset_name}' a partir do pipeline {pipeline.id}"
        )

        # Coletar chunks e análises
        chunks_with_analysis = self._collect_pipeline_content(
            pipeline=pipeline,
            categories=categories,
            content_types=content_types,
            additional_filters=additional_filters,
        )

        # Filtrar por relevância
        chunks_with_analysis = [
            (chunk, analysis)
            for chunk, analysis in chunks_with_analysis
            if analysis.relevance_score >= self.min_relevance_score
        ]

        logger.info(
            f"Coletados {len(chunks_with_analysis)} chunks relevantes (pontuação ≥ {self.min_relevance_score})"
        )

        # Processar e formatar exemplos
        examples = self._format_examples(
            chunks_with_analysis=chunks_with_analysis,
            format_type=format_type,
            max_length=max_length,
            include_metadata=include_metadata,
        )

        logger.info(f"Gerados {len(examples)} exemplos para o dataset")

        # Dividir em conjuntos de treino, validação e teste
        train_examples, val_examples, test_examples = self._split_dataset(examples)

        logger.info(
            f"Dataset dividido: {len(train_examples)} treino, {len(val_examples)} validação, {len(test_examples)} teste"
        )

        # Salvar dataset
        dataset_info = self._save_dataset(
            dataset_dir=dataset_dir,
            train_examples=train_examples,
            val_examples=val_examples,
            test_examples=test_examples,
            metadata={
                "name": dataset_name,
                "pipeline_id": pipeline.id,
                "pipeline_name": pipeline.name,
                "format_type": format_type,
                "created_at": datetime.now().isoformat(),
                "examples_count": len(examples),
                "train_count": len(train_examples),
                "val_count": len(val_examples),
                "test_count": len(test_examples),
                "min_relevance_score": self.min_relevance_score,
                "categories": [c.value for c in categories] if categories else None,
                "content_types": (
                    [t.value for t in content_types] if content_types else None
                ),
                "additional_filters": additional_filters,
            },
        )

        logger.info(f"Dataset '{dataset_name}' construído e salvo em {dataset_dir}")
        return dataset_info

    def load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Carrega um dataset existente.

        Args:
            dataset_name: Nome do dataset

        Returns:
            Informações e dados do dataset
        """
        dataset_dir = os.path.join(self.output_dir, dataset_name)

        if not os.path.exists(dataset_dir):
            logger.error(f"Dataset não encontrado: {dataset_name}")
            raise ValueError(f"Dataset não encontrado: {dataset_name}")

        # Carregar metadados
        metadata_path = os.path.join(dataset_dir, "metadata.json")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Carregar splits
        train_path = os.path.join(dataset_dir, "train.json")
        val_path = os.path.join(dataset_dir, "validation.json")
        test_path = os.path.join(dataset_dir, "test.json")

        with open(train_path, "r", encoding="utf-8") as f:
            train_examples = json.load(f)

        with open(val_path, "r", encoding="utf-8") as f:
            val_examples = json.load(f)

        with open(test_path, "r", encoding="utf-8") as f:
            test_examples = json.load(f)

        logger.info(
            f"Dataset '{dataset_name}' carregado: {len(train_examples)} treino, {len(val_examples)} validação, {len(test_examples)} teste"
        )

        return {
            "metadata": metadata,
            "train": train_examples,
            "validation": val_examples,
            "test": test_examples,
        }

    def export_to_hf_format(
        self, dataset_name: str, output_dir: Optional[str] = None
    ) -> str:
        """
        Exporta um dataset para o formato Hugging Face datasets.

        Args:
            dataset_name: Nome do dataset
            output_dir: Diretório de saída opcional

        Returns:
            Caminho para o dataset exportado
        """
        # Verificar dependências
        try:
            import datasets
        except ImportError:
            logger.error(
                "Biblioteca 'datasets' não encontrada. Instale com: pip install datasets"
            )
            raise ImportError(
                "Biblioteca 'datasets' necessária para exportar no formato Hugging Face"
            )

        # Carregar dataset
        dataset_data = self.load_dataset(dataset_name)

        # Determinar diretório de saída
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"{dataset_name}_hf")

        os.makedirs(output_dir, exist_ok=True)

        # Converter para formato datasets
        ds = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_list(dataset_data["train"]),
                "validation": datasets.Dataset.from_list(dataset_data["validation"]),
                "test": datasets.Dataset.from_list(dataset_data["test"]),
            }
        )

        # Salvar no formato Arrow
        ds.save_to_disk(output_dir)

        # Salvar também metadados em formato JSON
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(dataset_data["metadata"], f, indent=2, ensure_ascii=False)

        logger.info(
            f"Dataset '{dataset_name}' exportado no formato Hugging Face para {output_dir}"
        )
        return output_dir

    def _collect_pipeline_content(
        self,
        pipeline: Pipeline,
        categories: Optional[List[ContentCategory]] = None,
        content_types: Optional[List[ContentType]] = None,
        additional_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[ContentChunk, AnalysisResult]]:
        """
        Coleta chunks e análises de um pipeline com filtragem opcional.
        """
        content_chunks = []

        # Buscar todos os chunks e suas análises
        for chunk_ref in pipeline.content_chunks:
            chunk_id = chunk_ref.chunk_id
            analysis_id = chunk_ref.analysis_id

            # Carregar chunk e análise
            chunk = load_content_chunk(chunk_id)
            analysis = load_analysis_result(analysis_id)

            if not chunk or not analysis:
                logger.warning(
                    f"Chunk {chunk_id} ou análise {analysis_id} não encontrado(a)"
                )
                continue

            # Aplicar filtros se especificados
            if categories and chunk_ref.category not in categories:
                continue

            if content_types and chunk.content_type not in content_types:
                continue

            # Aplicar filtros adicionais
            if additional_filters:
                skip = False

                for key, value in additional_filters.items():
                    # Verificar filtros de análise
                    if key in analysis.__dict__:
                        if analysis.__dict__[key] != value:
                            skip = True
                            break

                    # Verificar filtros de chunk
                    elif key in chunk.__dict__:
                        if chunk.__dict__[key] != value:
                            skip = True
                            break

                    # Verificar filtros de metadata
                    elif key in chunk.metadata:
                        if chunk.metadata[key] != value:
                            skip = True
                            break

                if skip:
                    continue

            # Adicionar à lista final
            content_chunks.append((chunk, analysis))

        logger.info(f"Coletados {len(content_chunks)} chunks do pipeline {pipeline.id}")
        return content_chunks

    def _format_examples(
        self,
        chunks_with_analysis: List[Tuple[ContentChunk, AnalysisResult]],
        format_type: str,
        max_length: Optional[int] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Formata chunks e análises em exemplos para treinamento.

        Suporta diferentes formatos como instruction, completion, qa, etc.
        """
        examples = []

        for chunk, analysis in chunks_with_analysis:
            # Aplicar truncamento se especificado
            text = chunk.text
            if max_length and len(text) > max_length:
                text = text[:max_length]

            # Formatar com base no tipo selecionado
            if format_type == "instruction":
                # Formato: { "instruction": "...", "input": "...", "output": "..." }

                # Usar primeiras palavras como instrução
                instruction = f"Analise o seguinte texto: {text[:50]}..."

                # Entrada é o texto completo
                input_text = text

                # Saída é um resumo baseado na análise
                output = (
                    f"Este texto tem relevância de {analysis.relevance_score:.2f}. "
                )
                output += f"Palavras-chave: {', '.join(analysis.keywords[:5])}. "

                if analysis.summary:
                    output += f"Resumo: {analysis.summary}"

                example = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                }

            elif format_type == "completion":
                # Formato: { "text": "..." }
                example = {"text": text}

            elif format_type == "qa":
                # Formato: { "question": "...", "context": "...", "answer": "..." }

                # Gerar pergunta a partir do texto e análise
                keywords = analysis.keywords[:3]
                question = f"O que este texto diz sobre {', '.join(keywords)}?"

                # Contexto é o texto completo
                context = text

                # Resposta é derivada da análise
                answer = (
                    analysis.summary
                    if analysis.summary
                    else f"Este texto discute {', '.join(keywords)}."
                )

                example = {"question": question, "context": context, "answer": answer}

            elif format_type == "chat":
                # Formato: { "messages": [{"role": "...", "content": "..."}, ...] }

                # Criar conversa simulada com base no conteúdo
                messages = [
                    {
                        "role": "user",
                        "content": f"Pode analisar este texto para mim? {text[:100]}...",
                    },
                    {
                        "role": "assistant",
                        "content": f"Claro, analisarei o texto completo.",
                    },
                    {"role": "user", "content": text},
                    {
                        "role": "assistant",
                        "content": f"Análise: Este texto tem relevância de {analysis.relevance_score:.2f}. Principais palavras-chave: {', '.join(analysis.keywords[:5])}.",
                    },
                ]

                example = {"messages": messages}

            else:
                # Formato padrão: { "input_text": "...", "labels": "..." }
                example = {
                    "input_text": text,
                    "labels": {
                        "relevance": analysis.relevance_score,
                        "keywords": analysis.keywords,
                    },
                }

            # Adicionar metadados se solicitado
            if include_metadata:
                example["metadata"] = {
                    "chunk_id": chunk.id,
                    "source": chunk.source_location,
                    "content_type": (
                        chunk.content_type.value
                        if hasattr(chunk.content_type, "value")
                        else str(chunk.content_type)
                    ),
                    "created_at": (
                        chunk.created_at.isoformat()
                        if hasattr(chunk.created_at, "isoformat")
                        else str(chunk.created_at)
                    ),
                    "length": chunk.length,
                }

            # Gerar ID único para o exemplo
            example_id = hashlib.md5(
                f"{chunk.id}_{format_type}_{datetime.now().isoformat()}".encode()
            ).hexdigest()
            example["id"] = example_id

            examples.append(example)

        return examples

    def _split_dataset(
        self, examples: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Divide uma lista de exemplos em conjuntos de treino, validação e teste.
        """
        # Configurar semente para reprodutibilidade
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Embaralhar exemplos
        shuffled_examples = examples.copy()
        random.shuffle(shuffled_examples)

        # Calcular tamanhos dos conjuntos
        n_examples = len(shuffled_examples)
        n_test = max(1, int(n_examples * self.test_split))
        n_val = max(1, int(n_examples * self.val_split))
        n_train = n_examples - n_test - n_val

        # Garantir que conjuntos não estejam vazios
        if n_train <= 0:
            n_train = max(1, n_examples - 2)
            n_val = 1
            n_test = 1

        # Dividir exemplos
        train_examples = shuffled_examples[:n_train]
        val_examples = shuffled_examples[n_train : n_train + n_val]
        test_examples = shuffled_examples[n_train + n_val :]

        return train_examples, val_examples, test_examples

    def _save_dataset(
        self,
        dataset_dir: str,
        train_examples: List[Dict[str, Any]],
        val_examples: List[Dict[str, Any]],
        test_examples: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Salva o dataset em arquivos JSON.
        """
        # Salvar splits
        train_path = os.path.join(dataset_dir, "train.json")
        val_path = os.path.join(dataset_dir, "validation.json")
        test_path = os.path.join(dataset_dir, "test.json")
        metadata_path = os.path.join(dataset_dir, "metadata.json")

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_examples, f, indent=2, ensure_ascii=False)

        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_examples, f, indent=2, ensure_ascii=False)

        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_examples, f, indent=2, ensure_ascii=False)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Configurar caminhos no retorno
        metadata.update(
            {
                "paths": {
                    "train": train_path,
                    "validation": val_path,
                    "test": test_path,
                    "metadata": metadata_path,
                    "root": dataset_dir,
                }
            }
        )

        return metadata
