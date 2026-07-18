#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testes abrangentes para os analisadores do PipelineBuilder.

Este módulo contém testes para verificar a funcionalidade
dos diferentes analisadores implementados.
"""

import os
import sys
import uuid
import unittest
from datetime import datetime
from pathlib import Path
import mock
from unittest.mock import patch, MagicMock

# Adicionar diretório pai ao path para importação
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from cerberus_api.pipeline_builder.models import (
    DataSource,
    ContentChunk,
    AnalysisTask,
    SourceType,
    ContentType,
    TaskStatus,
    AnalysisResult,
)
from cerberus_api.pipeline_builder.tests.mock_analyzer import (
    MockAnalyzer,
    MockEnhancedAnalyzer,
)


class TestBaseAnalyzer:
    """
    Classe base para testes de analisadores.
    """

    def setUp(self):
        """
        Configuração para os testes.
        """
        # Criar diretório temporário para testes
        self.test_dir = Path("test_output/analyzers")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Criar tarefa de análise genérica
        self.task = AnalysisTask(
            id=str(uuid.uuid4()),
            chunk_ids=[str(uuid.uuid4()) for _ in range(3)],
            status=TaskStatus.PENDING,
            metadata={},
            created_at=datetime.now(),
        )

        # Criar chunks para análise
        self.chunks = [
            ContentChunk(
                id=self.task.chunk_ids[0],
                task_id=str(uuid.uuid4()),
                text="Este é um texto de exemplo para o primeiro chunk. " * 5,
                source_location="exemplo1.txt",
                content_type=ContentType.TEXT,
                length=200,
                metadata={},
                created_at=datetime.now(),
            ),
            ContentChunk(
                id=self.task.chunk_ids[1],
                task_id=str(uuid.uuid4()),
                text="Este é um texto diferente para o segundo chunk com mais detalhes. "
                * 5,
                source_location="exemplo2.txt",
                content_type=ContentType.TEXT,
                length=250,
                metadata={},
                created_at=datetime.now(),
            ),
            ContentChunk(
                id=self.task.chunk_ids[2],
                task_id=str(uuid.uuid4()),
                text="Este é o terceiro chunk com informações técnicas e detalhes específicos. "
                * 5,
                source_location="exemplo3.txt",
                content_type=ContentType.TEXT,
                length=300,
                metadata={},
                created_at=datetime.now(),
            ),
        ]

    def tearDown(self):
        """
        Limpeza após os testes.
        """
        # Opcionalmente remover diretório de testes
        # import shutil
        # shutil.rmtree(self.test_dir, ignore_errors=True)
        pass

    def assert_results_valid(self, results):
        """
        Verifica se os resultados da análise são válidos.
        """
        self.assertIsNotNone(results)
        self.assertIsInstance(results, list)

        if results:
            for result in results:
                self.assertIsNotNone(result.id)
                self.assertIsNotNone(result.task_id)
                self.assertIsNotNone(result.chunk_id)
                self.assertIn(result.chunk_id, self.task.chunk_ids)
                self.assertIsNotNone(result.created_at)
                self.assertIsInstance(result.metadata, dict)


class TestContentAnalyzer(unittest.TestCase, TestBaseAnalyzer):
    """
    Testes para o ContentAnalyzer.
    """

    def setUp(self):
        """
        Configuração para os testes do ContentAnalyzer.
        """
        TestBaseAnalyzer.setUp(self)
        self.analyzer = MockAnalyzer(
            storage_dir=str(self.test_dir / "content_analysis")
        )

    def test_basic_analysis(self):
        """
        Testa a análise básica de conteúdo.
        """
        # Executar análise
        results = self.analyzer.analyze(self.task, self.chunks)

        # Verificar resultados
        self.assert_results_valid(results)
        self.assertEqual(self.task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(self.task.completed_at)

        # Verificar número de resultados
        self.assertEqual(len(results), len(self.chunks))

        # Verificar conteúdo dos resultados
        for result in results:
            self.assertIn("word_count", result.metadata)
            self.assertIn("sentence_count", result.metadata)
            self.assertIn("keywords", result.metadata)
            self.assertIn("language", result.metadata)

            # Verificar se as palavras-chave são uma lista
            self.assertIsInstance(result.metadata["keywords"], list)
            self.assertGreater(len(result.metadata["keywords"]), 0)

    def test_analysis_with_empty_chunks(self):
        """
        Testa a análise com chunks vazios.
        """
        # Criar chunks vazios
        empty_chunks = [
            ContentChunk(
                id=str(uuid.uuid4()),
                task_id=str(uuid.uuid4()),
                text="",
                source_location="vazio.txt",
                content_type=ContentType.TEXT,
                length=0,
                metadata={},
                created_at=datetime.now(),
            )
        ]

        # Atualizar task para refletir os novos chunks
        self.task.chunk_ids = [chunk.id for chunk in empty_chunks]

        # Executar análise
        results = self.analyzer.analyze(self.task, empty_chunks)

        # Verificar resultados
        self.assertEqual(self.task.status, TaskStatus.COMPLETED)
        self.assertEqual(len(results), len(empty_chunks))

        # Verificar conteúdo dos resultados
        for result in results:
            self.assertEqual(result.metadata["word_count"], 0)
            self.assertEqual(result.metadata["sentence_count"], 0)
            self.assertGreaterEqual(len(result.metadata["keywords"]), 1)

    def test_analysis_with_custom_nlp(self):
        """
        Testa a análise com configuração personalizada de NLP.
        """
        # Configurar analisador com opções personalizadas
        analyzer = MockAnalyzer(
            storage_dir=str(self.test_dir / "content_analysis"),
            nlp_model="pt_core_news_lg",
            extract_entities=True,
            extract_sentiment=True,
        )

        # Executar análise
        results = analyzer.analyze(self.task, self.chunks)

        # Verificar resultados
        self.assert_results_valid(results)

        # Verificar conteúdo específico
        for result in results:
            self.assertEqual(result.metadata["language"], "portuguese")
            self.assertIn("entities", result.metadata)
            self.assertIn("sentiment", result.metadata)


class TestEnhancedAnalyzer(unittest.TestCase, TestBaseAnalyzer):
    """
    Testes para o EnhancedAnalyzer.
    """

    def setUp(self):
        """
        Configuração para os testes do EnhancedAnalyzer.
        """
        TestBaseAnalyzer.setUp(self)
        self.analyzer = MockEnhancedAnalyzer(
            storage_dir=str(self.test_dir / "enhanced_analysis")
        )

    def test_embedding_analysis(self):
        """
        Testa a análise com embeddings.
        """
        # Executar análise
        results = self.analyzer.analyze(self.task, self.chunks)

        # Verificar resultados
        self.assert_results_valid(results)
        self.assertEqual(self.task.status, TaskStatus.COMPLETED)

        # Verificar número de resultados
        self.assertEqual(len(results), len(self.chunks))

        # Verificar conteúdo específico
        for result in results:
            self.assertIn("embedding", result.metadata)
            self.assertIn("embedding_model", result.metadata)

    def test_analysis_with_similarity(self):
        """
        Testa a análise com cálculo de similaridade.
        """
        # Configurar analisador com cálculo de similaridade
        analyzer = MockEnhancedAnalyzer(
            storage_dir=str(self.test_dir / "enhanced_analysis"),
            calculate_similarity=True,
        )

        # Executar análise
        results = analyzer.analyze(self.task, self.chunks)

        # Verificar resultados
        self.assert_results_valid(results)

        # Verificar conteúdo específico
        for result in results:
            self.assertIn("similarity_scores", result.metadata)
            self.assertIsInstance(result.metadata["similarity_scores"], dict)

    def test_analysis_with_summarization(self):
        """
        Testa a análise com geração de resumo.
        """
        # Configurar analisador com geração de resumo
        analyzer = MockEnhancedAnalyzer(
            storage_dir=str(self.test_dir / "enhanced_analysis"), generate_summary=True
        )

        # Executar análise
        results = analyzer.analyze(self.task, self.chunks)

        # Verificar resultados
        self.assert_results_valid(results)

        # Verificar conteúdo específico
        for result in results:
            self.assertIn("summary", result.metadata)
            self.assertEqual(
                result.metadata["summary"],
                "Este é um resumo gerado automaticamente do texto.",
            )


if __name__ == "__main__":
    unittest.main()
