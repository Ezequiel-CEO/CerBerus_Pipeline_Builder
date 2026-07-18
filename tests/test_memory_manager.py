#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testes para o MemoryManager.
Verifica as funcionalidades de controle de memória e otimização.
"""

import unittest
import os
import sys
import tempfile
import logging
import time
from typing import Dict, Any, List

# Configurar caminho para módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from cerberus_api.pipeline_builder.utils.memory_manager import (
    MemoryManager,
    memory_manager,
)
from cerberus_api.pipeline_builder.models import ContentChunk, AnalysisResult
from cerberus_api.utils.logging_config import APILogger, get_logger

# Configurar logging para testes
logger = get_logger("test_memory_manager")


class TestMemoryManager(unittest.TestCase):
    """Testes para o MemoryManager."""

    def setUp(self):
        """Configuração para cada teste."""
        # Criar uma instância de teste separada
        self.test_manager = MemoryManager(
            ram_threshold=0.95,  # Limiar alto para não acionar limpeza
            vram_threshold=0.95,  # Limiar alto para não acionar offload
            enable_cuda=True,  # Testar com CUDA se disponível
            enable_streaming=True,  # Testar com streaming
            max_batch_size=8,  # Tamanho pequeno para testes
            log_interval=1,  # Intervalo curto para testes
        )

    def test_memory_detection(self):
        """Testa detecção de memória e dispositivos."""
        # Verificar configuração do dispositivo
        self.assertIn(self.test_manager.device, ["cpu", "cuda", "mps"])

        # Verificar se detect_cuda foi executado
        self.assertIsNotNone(self.test_manager.has_cuda)
        self.assertIsNotNone(self.test_manager.has_gpu)

        logger.info(f"Dispositivo detectado: {self.test_manager.device}")
        logger.info(f"CUDA disponível: {self.test_manager.has_cuda}")
        logger.info(f"GPU disponível: {self.test_manager.has_gpu}")

    def test_memory_usage_reporting(self):
        """Testa relatório de uso de memória."""
        # Obter uso de memória
        memory_info = self.test_manager.get_memory_usage()

        # Verificar se tem informações básicas
        self.assertIn("ram_used_mb", memory_info)
        self.assertIn("ram_total_mb", memory_info)
        self.assertIn("ram_percent", memory_info)

        # Verificar se valores são números positivos
        self.assertGreater(memory_info["ram_used_mb"], 0)
        self.assertGreater(memory_info["ram_total_mb"], 0)
        self.assertGreaterEqual(memory_info["ram_percent"], 0)
        self.assertLessEqual(memory_info["ram_percent"], 1.0)

        # Verificar log de memória
        log_result = self.test_manager.log_memory_usage("teste")
        self.assertIsInstance(log_result, dict)

    def test_batch_size_optimization(self):
        """Testa otimização de tamanho de lote."""
        # Testar para diferentes tamanhos de dados
        small_batch = self.test_manager.optimize_batch_size(10)
        medium_batch = self.test_manager.optimize_batch_size(100)
        large_batch = self.test_manager.optimize_batch_size(1000)

        # Verificar se os valores são razoáveis
        self.assertGreaterEqual(small_batch, 1)
        self.assertGreaterEqual(medium_batch, 1)
        self.assertGreaterEqual(large_batch, 1)

        # Verificar limites máximos
        self.assertLessEqual(small_batch, self.test_manager.max_batch_size)
        self.assertLessEqual(medium_batch, self.test_manager.max_batch_size)
        self.assertLessEqual(large_batch, self.test_manager.max_batch_size)

        logger.info(
            f"Tamanhos de lote otimizados: pequeno={small_batch}, médio={medium_batch}, grande={large_batch}"
        )

    def test_embedding_cache(self):
        """Testa cache de embeddings."""
        # Criar alguns embeddings de teste
        test_embedding1 = [0.1, 0.2, 0.3]
        test_embedding2 = [0.4, 0.5, 0.6]

        # Armazenar no cache
        self.test_manager.cache_embedding("teste1", test_embedding1)
        self.test_manager.cache_embedding("teste2", test_embedding2)

        # Recuperar do cache
        cached1 = self.test_manager.get_cached_embedding("teste1")
        cached2 = self.test_manager.get_cached_embedding("teste2")
        cached3 = self.test_manager.get_cached_embedding("nao_existe")

        # Verificar valores
        self.assertEqual(cached1, test_embedding1)
        self.assertEqual(cached2, test_embedding2)
        self.assertIsNone(cached3)

        # Testar limpeza de cache
        self.test_manager.clear_embedding_cache()
        self.assertIsNone(self.test_manager.get_cached_embedding("teste1"))

    def test_streaming_decision(self):
        """Testa decisão de processamento em streaming."""
        # Criar dados de teste de diferentes tamanhos
        small_data = 100 * 1024  # 100 KB
        small_items = [1024] * 10  # 10 items de 1 KB

        medium_data = 10 * 1024 * 1024  # 10 MB
        medium_items = [100 * 1024] * 10  # 10 itens de 100 KB

        large_data = 500 * 1024 * 1024  # 500 MB
        large_items = [1024 * 1024] * 100  # 100 itens de 1 MB

        mixed_items = [1024] * 90 + [
            50 * 1024 * 1024
        ] * 2  # 90 pequenos + 2 grandes (50MB)
        mixed_data = sum(mixed_items)

        # Verificar decisões
        small_decision = self.test_manager.should_process_in_streaming(
            small_data, small_items
        )
        medium_decision = self.test_manager.should_process_in_streaming(
            medium_data, medium_items
        )
        large_decision = self.test_manager.should_process_in_streaming(
            large_data, large_items
        )
        mixed_decision = self.test_manager.should_process_in_streaming(
            mixed_data, mixed_items
        )

        # Registrar resultados para inspeção
        logger.info(
            f"Decisão streaming - pequeno ({small_data/1024/1024:.2f}MB): {small_decision}"
        )
        logger.info(
            f"Decisão streaming - médio ({medium_data/1024/1024:.2f}MB): {medium_decision}"
        )
        logger.info(
            f"Decisão streaming - grande ({large_data/1024/1024:.2f}MB): {large_decision}"
        )
        logger.info(
            f"Decisão streaming - misto ({mixed_data/1024/1024:.2f}MB, 2 itens grandes): {mixed_decision}"
        )

        # Grandes volumes devem usar streaming
        self.assertTrue(large_decision)

        # Volumes mistos com itens grandes devem usar streaming
        self.assertTrue(mixed_decision)


class TestMemoryManagerIntegration(unittest.TestCase):
    """Testes de integração para o MemoryManager."""

    def test_global_instance(self):
        """Testa se a instância global está funcionando corretamente."""
        # Verificar se a instância global existe
        self.assertIsInstance(memory_manager, MemoryManager)

        # Testar funcionalidade básica
        memory_info = memory_manager.get_memory_usage()
        self.assertIn("ram_used_mb", memory_info)

        # Verificar limpeza de memória
        memory_manager.clear_memory()

        # Testar cache
        test_embedding = [0.1, 0.2, 0.3]
        memory_manager.cache_embedding("test_global", test_embedding)
        cached = memory_manager.get_cached_embedding("test_global")
        self.assertEqual(cached, test_embedding)


if __name__ == "__main__":
    unittest.main()
