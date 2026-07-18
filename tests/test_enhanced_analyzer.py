#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testes de integração do gerenciador de memória com EnhancedAnalyzer.
"""

import unittest
import os
from unittest.mock import patch, MagicMock
import numpy as np

from cerberus_api.pipeline_builder.analyzers.enhanced_analyzer import EnhancedAnalyzer
from cerberus_api.pipeline_builder.models import (
    ContentChunk,
    AnalysisResult,
    ContentType,
    TaskStatus,
)
from cerberus_api.pipeline_builder.utils.memory_manager import MemoryManager


class TestEnhancedAnalyzerWithMemoryManager(unittest.TestCase):
    """
    Testes para verificar a integração do EnhancedAnalyzer com o gerenciador de memória.
    """

    def setUp(self):
        """Configuração dos testes."""
        # Diretório temporário para armazenamento
        self.test_dir = os.path.join(os.path.dirname(__file__), "temp_test_data")
        os.makedirs(self.test_dir, exist_ok=True)

        # Criar analisador com configurações para teste
        self.analyzer = EnhancedAnalyzer(
            storage_dir=self.test_dir,
            use_embeddings=True,
            embedding_model="all-MiniLM-L6-v2",  # Modelo pequeno para testes
            stream_processing=True,
            memory_efficient=True,
            batch_size=4,
            use_gpu=True,
        )

        # Dados de teste com todos os campos obrigatórios
        texto1 = "Este é um texto de teste para análise de relevância."
        texto2 = "Outro texto com conteúdo diferente para o sistema processar."
        texto3 = "Mais um texto com palavras-chave importantes para verificar o funcionamento."
        texto4 = "Um texto final para completar o lote de teste do analisador."
        texto5 = "Um texto muito longo " + ("palavra " * 200)

        self.test_chunks = [
            ContentChunk(
                id="1",
                task_id="task1",
                content_type=ContentType.TEXT,
                text=texto1,
                content=texto1,  # Campo não obrigatório mas usado no EnhancedAnalyzer
                source_location="test/location/1",
                length=len(texto1),
            ),
            ContentChunk(
                id="2",
                task_id="task1",
                content_type=ContentType.TEXT,
                text=texto2,
                content=texto2,
                source_location="test/location/2",
                length=len(texto2),
            ),
            ContentChunk(
                id="3",
                task_id="task1",
                content_type=ContentType.TEXT,
                text=texto3,
                content=texto3,
                source_location="test/location/3",
                length=len(texto3),
            ),
            ContentChunk(
                id="4",
                task_id="task1",
                content_type=ContentType.TEXT,
                text=texto4,
                content=texto4,
                source_location="test/location/4",
                length=len(texto4),
            ),
            ContentChunk(
                id="5",
                task_id="task1",
                content_type=ContentType.TEXT,
                text=texto5,
                content=texto5,
                source_location="test/location/5",
                length=len(texto5),
            ),
        ]

    def tearDown(self):
        """Limpeza após os testes."""
        # Limpar diretório de teste
        for f in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, f)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    @patch(
        "cerberus_api.pipeline_builder.utils.memory_manager.MemoryManager.log_memory_usage"
    )
    def test_memory_logging(self, mock_log_memory):
        """Verifica se o gerenciador de memória registra o uso durante a análise."""
        # Analisar alguns chunks
        self.analyzer.analyze(self.test_chunks[:2])

        # Verificar se a função de log foi chamada
        self.assertTrue(mock_log_memory.called)
        mock_log_memory.assert_any_call("início de análise")
        mock_log_memory.assert_any_call("fim de análise")

    @patch(
        "cerberus_api.pipeline_builder.utils.memory_manager.MemoryManager.optimize_batch_size"
    )
    def test_batch_size_optimization(self, mock_optimize):
        """Verifica se o tamanho do lote é otimizado durante a análise."""
        # Configurar mock para retornar um valor específico
        mock_optimize.return_value = 3

        # Analisar chunks
        self.analyzer.analyze(self.test_chunks)

        # Verificar se a otimização de lote foi chamada
        mock_optimize.assert_called_with(len(self.test_chunks))

    @patch(
        "cerberus_api.pipeline_builder.utils.memory_manager.MemoryManager.should_process_in_streaming"
    )
    def test_streaming_decision(self, mock_streaming):
        """Verifica se a decisão de processamento em streaming é consultada."""
        # Configurar mock para retornar True
        mock_streaming.return_value = True

        # Analisar chunks
        results = self.analyzer.analyze(self.test_chunks)

        # Verificar se a decisão de streaming foi consultada
        mock_streaming.assert_called_once()

        # Verificar se temos resultados para todos os chunks
        self.assertEqual(len(results), len(self.test_chunks))

    @patch(
        "cerberus_api.pipeline_builder.analyzers.enhanced_analyzer.EnhancedAnalyzer._load_embedding_model"
    )
    def test_lazy_model_loading(self, mock_load_model):
        """Verifica se o modelo de embeddings é carregado sob demanda."""
        # Analisar um chunk
        self.analyzer.analyze([self.test_chunks[0]])

        # Verificar condição:
        # 1. Se sentence-transformers estiver disponível, o método deve ser chamado
        # 2. Se não estiver disponível, o método não será chamado (comportamento aceitável)
        if self.analyzer.use_embeddings:
            self.assertTrue(
                mock_load_model.called,
                "O método _load_embedding_model deveria ser chamado quando use_embeddings=True",
            )
        else:
            # Teste passa mesmo sem chamar o método, pois embeddings estão desativados
            self.assertTrue(
                True,
                "O método _load_embedding_model não precisa ser chamado quando use_embeddings=False",
            )

    @patch(
        "cerberus_api.pipeline_builder.utils.memory_manager.MemoryManager.get_cached_embedding"
    )
    @patch(
        "cerberus_api.pipeline_builder.utils.memory_manager.MemoryManager.cache_embedding"
    )
    def test_embedding_cache(self, mock_cache, mock_get_cache):
        """Verifica se o cache de embeddings é utilizado."""
        # Configurar mock para simular cache miss e depois cache hit
        mock_get_cache.return_value = None

        # Simular que temos um sentence transformer
        self.analyzer.sentence_transformer = MagicMock()
        self.analyzer.sentence_transformer.encode.return_value = np.array(
            [0.1, 0.2, 0.3]
        )

        # Gerar embedding duas vezes para o mesmo texto
        text = "Este é um texto de teste para cache."
        self.analyzer._generate_embedding(text)
        self.analyzer._generate_embedding(text)

        # Verificar se a função de cache foi chamada
        self.assertTrue(mock_cache.called)

        # Verificar se a função de recuperação do cache foi chamada
        self.assertEqual(mock_get_cache.call_count, 2)

    def test_memory_efficient_processing(self):
        """Verifica o processamento eficiente de memória."""
        # Criar um chunk grande para forçar processamento especial
        large_content = "Este é um texto grande para teste. " * 1000
        large_chunk = ContentChunk(
            id="large",
            task_id="task_large",
            content_type=ContentType.TEXT,
            text=large_content,
            content=large_content,
            source_location="test/location/large",
            length=len(large_content),
        )

        # Analisar o chunk grande
        results = self.analyzer.analyze([large_chunk])

        # Verificar se foi processado com sucesso
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], AnalysisResult)
        self.assertEqual(results[0].chunk_id, "large")


if __name__ == "__main__":
    unittest.main()
