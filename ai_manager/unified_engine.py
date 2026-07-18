#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UnifiedEngine - Motor Unificado do CerBerusFMK
============================================

Combina o melhor dos dois mundos:
- Performance do motor nativo C++/CUDA
- Flexibilidade da interface Python de alto nível
- Sistema de fallback inteligente
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from cerberus_api.utils.logging_config import get_logger
from cerberus_api.pipeline_builder.utils.memory_manager import MemoryManager

# Importar implementação nativa
try:
    from libs.CerBerusFMK.CerBerusFMK.inference_engine.engine import InferenceEngine
    from libs.CerBerusFMK.CerBerusFMK.llm_local.native_gguf import NativeGGUFRunner

    NATIVE_ENGINE_AVAILABLE = True
except ImportError:
    NATIVE_ENGINE_AVAILABLE = False

# Importar implementação Python
try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

logger = get_logger("unified_engine")


class UnifiedEngine:
    """
    Motor unificado do CerBerusFMK que combina as melhores características
    dos motores nativo e Python.
    """

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = True,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        **kwargs,
    ):
        """
        Inicializa o motor unificado.

        Args:
            model_path: Caminho para o modelo
            use_gpu: Se deve usar GPU quando disponível
            n_gpu_layers: Número de camadas para processar na GPU (-1 = todas)
            n_ctx: Tamanho do contexto em tokens
            **kwargs: Parâmetros adicionais
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.kwargs = kwargs

        # Estado interno
        self.model = None
        self.engine_type = None
        self.is_loaded = False
        self.memory_manager = MemoryManager()

        # Métricas
        self.metrics = {"total_tokens": 0, "total_time": 0.0, "inference_count": 0}

        # Tentar carregar o modelo
        self._initialize()

    def _initialize(self):
        """Inicializa o motor com a melhor implementação disponível."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")

        # Verificar tipo do modelo
        model_type = self._detect_model_type()

        # Tentar usar motor nativo primeiro
        if NATIVE_ENGINE_AVAILABLE and self.use_gpu:
            try:
                logger.info("Tentando usar motor nativo C++/CUDA...")
                self.model = InferenceEngine()
                self.model.load_model(self.model_path)
                self.engine_type = "native"
                self.is_loaded = True
                logger.info("Motor nativo C++/CUDA inicializado com sucesso")
                return
            except Exception as e:
                logger.warning(f"Não foi possível usar motor nativo: {e}")

        # Fallback para implementação Python
        if LLAMA_CPP_AVAILABLE:
            try:
                logger.info("Usando implementação Python (llama-cpp)...")
                self.model = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=self.n_gpu_layers if self.use_gpu else 0,
                    n_ctx=self.n_ctx,
                    **self.kwargs,
                )
                self.engine_type = "python"
                self.is_loaded = True
                logger.info("Motor Python (llama-cpp) inicializado com sucesso")
                return
            except Exception as e:
                logger.error(f"Não foi possível inicializar llama-cpp: {e}")

        raise RuntimeError("Nenhuma implementação disponível para carregar o modelo")

    def _detect_model_type(self) -> str:
        """Detecta o tipo do modelo pelo nome do arquivo."""
        if self.model_path.endswith(".gguf"):
            return "gguf"
        elif self.model_path.endswith(".ggml"):
            return "ggml"
        elif self.model_path.endswith(".safetensors"):
            return "safetensors"
        else:
            return "unknown"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Gera texto usando o modelo carregado.

        Args:
            prompt: Texto de entrada
            max_tokens: Número máximo de tokens a gerar
            temperature: Temperatura para sampling
            top_p: Valor para nucleus sampling
            top_k: Valor para top-k sampling
            stop: Lista de sequências para parar geração
            **kwargs: Parâmetros adicionais

        Returns:
            Tuple[str, Dict]: (texto gerado, metadados)
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo não está carregado")

        start_time = time.time()

        try:
            if self.engine_type == "native":
                # Usar implementação nativa
                output = self.model.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop_sequences=stop,
                )
            else:
                # Usar implementação Python
                output = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop,
                    **kwargs,
                )

            # Atualizar métricas
            elapsed = time.time() - start_time
            self.metrics["total_time"] += elapsed
            self.metrics["inference_count"] += 1
            self.metrics["total_tokens"] += len(output["choices"][0]["text"].split())

            return output["choices"][0]["text"], {
                "time": elapsed,
                "engine": self.engine_type,
                **output.get("usage", {}),
            }

        except Exception as e:
            logger.error(f"Erro durante geração: {e}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de uso do motor."""
        return {
            **self.metrics,
            "engine_type": self.engine_type,
            "avg_tokens_per_sec": (
                self.metrics["total_tokens"] / self.metrics["total_time"]
                if self.metrics["total_time"] > 0
                else 0
            ),
        }

    def __del__(self):
        """Cleanup ao destruir o objeto."""
        if self.model:
            try:
                if self.engine_type == "native":
                    self.model.unload()
                # llama-cpp limpa automaticamente
            except:
                pass
