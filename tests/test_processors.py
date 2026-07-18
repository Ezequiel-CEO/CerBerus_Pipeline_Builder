#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testes abrangentes para os processadores do PipelineBuilder.

Este módulo contém testes para verificar a funcionalidade
dos diferentes processadores implementados.
"""

import os
import sys
import uuid
import json
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Adicionar diretório pai ao path para importação
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from cerberus_api.pipeline_builder.models import (
    DataSource,
    ContentChunk,
    AnalysisTask,
    AnalysisResult,
    CollectionTask,
    ProcessingTask,
    ProcessingResult,
    Pipeline,
    PipelineStage,
    PipelineStep,
    SourceType,
    ContentType,
    TaskStatus,
)
from cerberus_api.pipeline_builder.processors import PipelineProcessor


class TestPipelineProcessor(unittest.TestCase):
    """
    Testes para o PipelineProcessor.
    """

    def setUp(self):
        """
        Configuração para os testes.
        """
        # Criar diretório temporário para testes
        self.test_dir = Path("test_output/processors")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Criar processador
        self.processor = PipelineProcessor(
            storage_dir=str(self.test_dir / "pipeline_processing")
        )

        # Criar fontes de dados de teste
        self.sources = [
            DataSource(
                id=str(uuid.uuid4()),
                name="Fonte Web de Teste",
                source_type=SourceType.WEB,
                location="https://example.com/test",
                description="Fonte web para testes do processador",
                metadata={},
                created_at=datetime.now(),
            ),
            DataSource(
                id=str(uuid.uuid4()),
                name="Fonte PDF de Teste",
                source_type=SourceType.PDF,
                location="https://example.com/test.pdf",
                description="Fonte PDF para testes do processador",
                metadata={},
                created_at=datetime.now(),
            ),
        ]

        # Criar chunks de conteúdo de teste
        self.chunks = []
        for i in range(5):
            self.chunks.append(
                ContentChunk(
                    id=str(uuid.uuid4()),
                    task_id=str(uuid.uuid4()),
                    text=f"Conteúdo de teste para o chunk {i+1}. " * 10,
                    source_location=f"fonte{i+1}.txt",
                    content_type=ContentType.TEXT,
                    length=300,
                    metadata={},
                    created_at=datetime.now(),
                )
            )

        # Criar resultados de análise de teste
        self.analysis_results = []
        for chunk in self.chunks:
            self.analysis_results.append(
                AnalysisResult(
                    id=str(uuid.uuid4()),
                    task_id=str(uuid.uuid4()),
                    chunk_id=chunk.id,
                    metadata={
                        "word_count": 50,
                        "keywords": ["teste", "processador", "pipeline"],
                        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                        "relevance_score": 0.85,
                    },
                    created_at=datetime.now(),
                )
            )

        # Criar pipeline de teste
        self.pipeline = Pipeline(
            id=str(uuid.uuid4()),
            name="Pipeline de Teste",
            description="Pipeline para testes do processador",
            stages=[
                PipelineStage(
                    id=str(uuid.uuid4()),
                    name="Estágio de Coleta",
                    description="Estágio para coletar dados",
                    steps=[
                        PipelineStep(
                            id=str(uuid.uuid4()),
                            name="Coleta Web",
                            description="Coleta de dados da web",
                            processor_type="WebCollector",
                            parameters={},
                            source_ids=[self.sources[0].id],
                        )
                    ],
                ),
                PipelineStage(
                    id=str(uuid.uuid4()),
                    name="Estágio de Análise",
                    description="Estágio para analisar conteúdo",
                    steps=[
                        PipelineStep(
                            id=str(uuid.uuid4()),
                            name="Análise de Conteúdo",
                            description="Análise do conteúdo coletado",
                            processor_type="ContentAnalyzer",
                            parameters={},
                            source_ids=[],
                        )
                    ],
                ),
            ],
            metadata={},
            created_at=datetime.now(),
        )

        # Criar tarefa de processamento de teste
        self.task = ProcessingTask(
            id=str(uuid.uuid4()),
            pipeline_id=self.pipeline.id,
            status=TaskStatus.PENDING,
            metadata={},
            created_at=datetime.now(),
        )

    def tearDown(self):
        """
        Limpeza após os testes.
        """
        # Opcionalmente remover diretório de testes
        # import shutil
        # shutil.rmtree(self.test_dir, ignore_errors=True)
        pass

    @patch("cerberus_api.pipeline_builder.processors.pipeline_processor.WebCollector")
    @patch(
        "cerberus_api.pipeline_builder.processors.pipeline_processor.ContentAnalyzer"
    )
    def test_process_pipeline(self, mock_analyzer, mock_collector):
        """
        Testa o processamento completo de um pipeline.
        """
        # Mock para WebCollector
        mock_collector_instance = MagicMock()
        mock_collector_instance.collect.return_value = self.chunks
        mock_collector.return_value = mock_collector_instance

        # Mock para ContentAnalyzer
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.return_value = self.analysis_results
        mock_analyzer.return_value = mock_analyzer_instance

        # Executar processamento
        results = self.processor.process(self.task, self.pipeline, self.sources)

        # Verificar resultados
        self.assertIsNotNone(results)
        self.assertIsInstance(results, list)
        self.assertEqual(self.task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(self.task.completed_at)

        # Verificar que o coletor e o analisador foram chamados
        mock_collector_instance.collect.assert_called()
        mock_analyzer_instance.analyze.assert_called()

    def test_create_processing_results(self):
        """
        Testa a criação de resultados de processamento.
        """
        # Criar resultados
        results = self.processor._create_processing_results(
            self.task, self.pipeline, self.chunks, self.analysis_results
        )

        # Verificar resultados
        self.assertIsNotNone(results)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(self.chunks))

        # Verificar conteúdo dos resultados
        for result in results:
            self.assertIsNotNone(result.id)
            self.assertEqual(result.task_id, self.task.id)
            self.assertEqual(result.pipeline_id, self.pipeline.id)
            self.assertIsNotNone(result.chunk_id)
            self.assertIn(result.chunk_id, [chunk.id for chunk in self.chunks])
            self.assertIsNotNone(result.created_at)
            self.assertIsInstance(result.metadata, dict)

    @patch("cerberus_api.pipeline_builder.processors.pipeline_processor.WebCollector")
    def test_process_with_empty_results(self, mock_collector):
        """
        Testa o processamento quando não há resultados de coleta.
        """
        # Mock para WebCollector retornando lista vazia
        mock_collector_instance = MagicMock()
        mock_collector_instance.collect.return_value = []
        mock_collector.return_value = mock_collector_instance

        # Executar processamento
        results = self.processor.process(self.task, self.pipeline, self.sources)

        # Verificar resultados
        self.assertEqual(len(results), 0)
        self.assertEqual(self.task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(self.task.completed_at)

        # Verificar metadados da tarefa
        self.assertIn("chunks_collected", self.task.metadata)
        self.assertEqual(self.task.metadata["chunks_collected"], 0)

    @patch("cerberus_api.pipeline_builder.processors.pipeline_processor.WebCollector")
    def test_process_with_collector_error(self, mock_collector):
        """
        Testa o processamento quando ocorre um erro no coletor.
        """
        # Mock para WebCollector lançando exceção
        mock_collector_instance = MagicMock()
        mock_collector_instance.collect.side_effect = Exception(
            "Erro de teste no coletor"
        )
        mock_collector.return_value = mock_collector_instance

        # Executar processamento
        results = self.processor.process(self.task, self.pipeline, self.sources)

        # Verificar resultados
        self.assertEqual(len(results), 0)
        self.assertEqual(self.task.status, TaskStatus.FAILED)
        self.assertIsNotNone(self.task.completed_at)

        # Verificar metadados da tarefa de erro
        self.assertIn("error", self.task.metadata)
        self.assertIn("Erro de teste no coletor", self.task.metadata["error"])

    @patch("cerberus_api.pipeline_builder.processors.pipeline_processor.WebCollector")
    @patch(
        "cerberus_api.pipeline_builder.processors.pipeline_processor.ContentAnalyzer"
    )
    def test_process_with_custom_parameters(self, mock_analyzer, mock_collector):
        """
        Testa o processamento com parâmetros personalizados.
        """
        # Mock para WebCollector
        mock_collector_instance = MagicMock()
        mock_collector_instance.collect.return_value = self.chunks
        mock_collector.return_value = mock_collector_instance

        # Mock para ContentAnalyzer
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.return_value = self.analysis_results
        mock_analyzer.return_value = mock_analyzer_instance

        # Atualizar pipeline com parâmetros personalizados
        custom_pipeline = self.pipeline
        custom_pipeline.stages[0].steps[0].parameters = {"max_depth": 2, "timeout": 10}
        custom_pipeline.stages[1].steps[0].parameters = {
            "extract_entities": True,
            "extract_sentiment": True,
        }

        # Executar processamento
        results = self.processor.process(self.task, custom_pipeline, self.sources)

        # Verificar que o coletor e o analisador foram chamados com os parâmetros corretos
        mock_collector.assert_called_with(
            storage_dir=unittest.mock.ANY, max_depth=2, timeout=10
        )
        mock_analyzer.assert_called_with(
            storage_dir=unittest.mock.ANY, extract_entities=True, extract_sentiment=True
        )


if __name__ == "__main__":
    unittest.main()
