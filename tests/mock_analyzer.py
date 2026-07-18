#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classes de mock para testes dos analisadores do PipelineBuilder.

Este módulo fornece implementações simuladas dos analisadores
para facilitar os testes unitários sem dependências externas.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from cerberus_api.pipeline_builder.models import (
    ContentChunk,
    AnalysisResult,
    AnalysisTask,
    TaskStatus,
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MockAnalyzer")


class MockAnalyzer:
    """
    Analisador mock para testes que simula funcionalidade do ContentAnalyzer.
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        nlp_model: str = "pt_core_news_sm",
        extract_entities: bool = False,
        extract_sentiment: bool = False,
    ):
        """
        Inicializa o analisador mock.

        Args:
            storage_dir: Diretório para salvar resultados de análise (opcional)
            nlp_model: Modelo de NLP para usar (não é usado de verdade, apenas simulado)
            extract_entities: Se deve extrair entidades (simulado)
            extract_sentiment: Se deve analisar sentimento (simulado)
        """
        self.storage_dir = storage_dir
        if storage_dir and not os.path.exists(storage_dir):
            os.makedirs(storage_dir, exist_ok=True)

        self.nlp_model = nlp_model
        self.extract_entities = extract_entities
        self.extract_sentiment = extract_sentiment
        logger.info(f"MockAnalyzer inicializado com modelo: {nlp_model}")

    def analyze(
        self, task: AnalysisTask, chunks: List[ContentChunk]
    ) -> List[AnalysisResult]:
        """
        Simula a análise de uma lista de chunks de conteúdo.

        Args:
            task: Tarefa de análise a ser processada
            chunks: Lista de chunks de conteúdo para analisar

        Returns:
            Lista de resultados de análise
        """
        logger.info(f"Analisando tarefa {task.id} com {len(chunks)} chunks")
        results = []

        for chunk in chunks:
            # Gerar metadados simulados
            metadata = self._generate_metadata(chunk.text)

            # Criar resultado
            result = AnalysisResult(
                id=str(uuid.uuid4()),
                task_id=task.id,
                chunk_id=chunk.id,
                metadata=metadata,
                created_at=datetime.now(),
            )

            results.append(result)

            # Salvar resultado se diretório de armazenamento estiver configurado
            if self.storage_dir:
                result_path = os.path.join(self.storage_dir, f"{result.id}.json")
                with open(result_path, "w") as f:
                    f.write(result.model_dump_json())

        # Atualizar status da tarefa
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()

        logger.info(f"Análise concluída para tarefa {task.id}")
        return results

    def _generate_metadata(self, text: str) -> Dict[str, Any]:
        """
        Gera metadados simulados para um texto.

        Args:
            text: Texto a ser analisado

        Returns:
            Dicionário de metadados
        """
        # Contar palavras e sentenças (simplificado)
        word_count = len(text.split()) if text else 0
        sentence_count = len([s for s in text.split(".") if s.strip()]) if text else 0

        # Metadados básicos
        metadata = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "language": "portuguese",
            "keywords": ["palavra1", "palavra2", "palavra3"],
            "relevance": 0.85,
        }

        # Adicionar entidades se solicitado
        if self.extract_entities:
            metadata["entities"] = [
                {"text": "Entidade1", "label": "PERSON", "start": 0, "end": 10},
                {"text": "Entidade2", "label": "ORG", "start": 20, "end": 30},
            ]

        # Adicionar sentimento se solicitado
        if self.extract_sentiment:
            metadata["sentiment"] = 0.75

        return metadata


class MockEnhancedAnalyzer(MockAnalyzer):
    """
    Analisador mock avançado que simula o EnhancedAnalyzer.
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        nlp_model: str = "pt_core_news_lg",
        extract_entities: bool = True,
        extract_sentiment: bool = True,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        calculate_similarity: bool = False,
        generate_summary: bool = False,
    ):
        """
        Inicializa o analisador mock avançado.

        Args:
            storage_dir: Diretório para salvar resultados de análise (opcional)
            nlp_model: Modelo de NLP para usar (simulado)
            extract_entities: Se deve extrair entidades (simulado)
            extract_sentiment: Se deve analisar sentimento (simulado)
            embedding_model: Modelo de embeddings para usar (simulado)
            calculate_similarity: Se deve calcular similaridade entre chunks
            generate_summary: Se deve gerar resumos dos textos
        """
        super().__init__(storage_dir, nlp_model, extract_entities, extract_sentiment)
        self.embedding_model = embedding_model
        self.calculate_similarity = calculate_similarity
        self.generate_summary = generate_summary
        logger.info(
            f"MockEnhancedAnalyzer inicializado com modelo de embedding: {embedding_model}"
        )

    def analyze(
        self, task: AnalysisTask, chunks: List[ContentChunk]
    ) -> List[AnalysisResult]:
        """
        Simula a análise avançada de uma lista de chunks de conteúdo.

        Args:
            task: Tarefa de análise a ser processada
            chunks: Lista de chunks de conteúdo para analisar

        Returns:
            Lista de resultados de análise
        """
        logger.info(f"Realizando análise avançada para tarefa {task.id}")

        # Chamar método de análise da classe base
        results = super().analyze(task, chunks)

        # Adicionar campos avançados a cada resultado
        for result in results:
            # Adicionar embedding simulado
            result.metadata["embedding"] = [0.1, 0.2, 0.3, 0.4, 0.5]
            result.metadata["embedding_model"] = self.embedding_model

            # Calcular similaridade se solicitado
            if self.calculate_similarity:
                similarity_scores = {}
                for chunk in chunks:
                    if chunk.id != result.chunk_id:
                        similarity_scores[chunk.id] = 0.75  # Valor simulado
                result.metadata["similarity_scores"] = similarity_scores

            # Gerar resumo se solicitado
            if self.generate_summary:
                result.metadata["summary"] = (
                    "Este é um resumo gerado automaticamente do texto."
                )

        return results
