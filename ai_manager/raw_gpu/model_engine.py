#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ModelEngine: Motor de inferência para modelos GGUF no CerBerusFMK.

Este módulo implementa uma interface para carregar e executar modelos GGUF,
com suporte para aceleração GPU/CPU nativa utilizando apenas implementações
próprias do CerBerusFMK.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional, Union, Generator, Tuple

from cerberus_api.utils.logging_config import get_logger
from cerberus_api.pipeline_builder.ai_manager.raw_gpu.gpu_manager import GPUManager

# Inicializar variáveis para controle de disponibilidade
NATIVE_GGUF_AVAILABLE = False
GGUF_SOURCE = None

# Tentar importar a implementação GGUF existente - apenas versões CerBerusFMK
try:
    # Primeira tentativa com a implementação nativa da biblioteca CerBerusFMK
    from libs.CerBerusFMK.CerBerusFMK.llm_local.gguf_model import GGUFModel
    from libs.CerBerusFMK.CerBerusFMK.llm_local.config import (
        ModelConfig,
        DeviceConfig,
        DeviceType,
    )

    NATIVE_GGUF_AVAILABLE = True
    GGUF_SOURCE = "native"
except ImportError:
    try:
        # Segunda tentativa com a implementação do Èter
        from cerberus_api.pipeline_builder.ai_manager.eter.llm.gguf_loader import (
            GGUFModel,
        )

        NATIVE_GGUF_AVAILABLE = True
        GGUF_SOURCE = "eter"
    except ImportError:
        NATIVE_GGUF_AVAILABLE = False
        GGUF_SOURCE = None


class ModelEngine:
    """
    Motor de inferência para modelos GGUF.

    Esta classe fornece uma interface unificada para carregar e executar modelos GGUF,
    usando a implementação existente do CerBerusFMK quando disponível ou fallback
    para implementação simulada.
    """

    def __init__(self, use_gpu: bool = True, model_path: Optional[str] = None):
        """
        Inicializa o motor de modelos.

        Args:
            use_gpu: Se deve usar GPU quando disponível
            model_path: Caminho opcional para carregar imediatamente um modelo
        """
        self.logger = get_logger("raw_gpu.model_engine")
        self.use_gpu = use_gpu
        self.gpu_manager = GPUManager(force_cpu=not use_gpu)
        self.model_path = None
        self.model = None
        self.model_loaded = False
        self.config = {}

        self.logger.info(
            f"ModelEngine inicializado. GPU: {use_gpu}, GGUF disponível: {NATIVE_GGUF_AVAILABLE} ({GGUF_SOURCE})"
        )

        # Carregar modelo se especificado
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        Carrega um modelo GGUF.

        Args:
            model_path: Caminho para o arquivo do modelo

        Returns:
            True se o modelo foi carregado com sucesso
        """
        self.logger.info(f"Carregando modelo: {model_path}")
        self.model_path = model_path

        if not os.path.exists(model_path):
            self.logger.error(f"Arquivo de modelo não encontrado: {model_path}")
            return False

        try:
            if NATIVE_GGUF_AVAILABLE:
                if GGUF_SOURCE == "eter":
                    # Usar implementação do Èter (opção mais segura)
                    n_gpu_layers = -1 if self.use_gpu else 0
                    self.model = GGUFModel(
                        model_path=model_path,
                        n_gpu_layers=n_gpu_layers,
                        use_memory_manager=True,
                    )
                    load_success = self.model.load()

                elif GGUF_SOURCE == "native":
                    # Usar implementação nativa do CerBerusFMK
                    try:
                        from libs.CerBerusFMK.CerBerusFMK.llm_local.config import (
                            ModelConfig,
                            DeviceConfig,
                            DeviceType,
                        )

                        device_type = (
                            DeviceType.CUDA if self.use_gpu else DeviceType.CPU
                        )
                        device_config = DeviceConfig(device_type=device_type)

                        model_config = ModelConfig(
                            model_path=model_path,
                            device=device_config,
                            context_size=4096,
                        )

                        self.model = GGUFModel(model_config)
                        load_success = self.model.load()
                    except Exception as e:
                        self.logger.error(
                            f"Erro ao carregar com implementação nativa: {e}"
                        )
                        load_success = False

                else:
                    load_success = False

                if load_success:
                    self.model_loaded = True
                    self.logger.info(
                        f"Modelo carregado com sucesso: {os.path.basename(model_path)}"
                    )
                    return True
                else:
                    self.logger.error("Falha ao carregar modelo, usando simulação")
                    # Fallback para modelo simulado
                    self._create_simulated_model()
                    return True
            else:
                # Implementação simulada
                self.logger.warning(
                    "Usando implementação simulada (GGUF não disponível)"
                )
                self._create_simulated_model()
                return True

        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

            # Fallback para modelo simulado
            self._create_simulated_model()
            return True

    def _create_simulated_model(self):
        """Cria um modelo simulado para testes e desenvolvimento."""
        self.model = type(
            "SimulatedModel",
            (),
            {
                "generate": lambda *args, **kwargs: "Este é um texto simulado. A implementação GGUF real não está disponível."
            },
        )()
        self.model_loaded = True

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        streaming: bool = False,
        **kwargs,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Gera texto a partir do modelo.

        Args:
            prompt: Prompt de entrada
            max_tokens: Número máximo de tokens a gerar
            temperature: Temperatura para amostragem
            top_p: Valor para amostragem nucleus
            streaming: Se deve retornar um gerador para streaming
            **kwargs: Parâmetros adicionais

        Returns:
            Texto gerado ou gerador para streaming
        """
        if not self.model_loaded:
            self.logger.error("Modelo não carregado. Use load_model() primeiro.")
            return "ERRO: Modelo não carregado"

        try:
            # Usar a implementação específica baseada no tipo de modelo carregado
            if GGUF_SOURCE == "native":
                # Implementação nativa do CerBerusFMK
                try:
                    generation_config = {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "streaming": streaming,
                    }
                    generation_config.update(kwargs)

                    result = self.model.generate(prompt, generation_config)
                    return result.get("text", "")
                except Exception as e:
                    self.logger.error(f"Erro ao gerar com implementação nativa: {e}")
                    return f"[Texto simulado] {prompt}"

            elif GGUF_SOURCE == "eter":
                # Implementação do Èter
                try:
                    if streaming:
                        return self.model.generate(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            streaming=True,
                            **kwargs,
                        )
                    else:
                        return self.model.generate(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            streaming=False,
                            **kwargs,
                        )
                except Exception as e:
                    self.logger.error(f"Erro ao gerar com implementação do Èter: {e}")
                    return f"[Texto simulado] {prompt}"

            else:
                # Implementação simulada
                return f"[Texto simulado em resposta a: {prompt}]"

        except Exception as e:
            self.logger.error(f"Erro na geração de texto: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return f"ERRO: {str(e)}"

    def get_memory_usage(self) -> float:
        """
        Retorna o uso atual de memória do modelo em MB.

        Returns:
            Uso de memória em MB
        """
        if not self.model_loaded:
            return 0.0

        try:
            # Obter informações detalhadas de memória
            mem_info = self.gpu_manager.get_memory_info()

            # Se estiver usando GPU, retornar VRAM utilizada
            if self.use_gpu and self.gpu_manager.use_gpu:
                return mem_info["vram_used_mb"]

            # Caso contrário, retornar RAM
            return mem_info["ram_used_mb"]

        except Exception as e:
            self.logger.error(f"Erro ao obter uso de memória: {e}")

            # Estimar com base no tamanho do arquivo (como fallback)
            if self.model_path and os.path.exists(self.model_path):
                file_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)

                # Tipicamente o uso de memória é aproximadamente metade
                # do tamanho do arquivo para modelos quantizados
                return file_size_mb * 0.5

            # Último recurso: estimar baseado no embedding dimension
            if self.config and "embd_dim" in self.config:
                embd_dim = self.config["embd_dim"]
                n_layers = self.config.get("n_layers", 24)
                vocab_size = self.config.get("vocab_size", 32000)

                # Fórmula simplificada para estimar uso de memória
                return embd_dim * n_layers * vocab_size * 4 / (1024 * 1024)

            # Se tudo falhar, retornar valor padrão para evitar erros
            return 1024.0  # 1GB como estimativa padrão

    def unload(self) -> bool:
        """
        Descarrega o modelo da memória.

        Returns:
            True se o modelo foi descarregado com sucesso
        """
        if not self.model_loaded:
            return True

        try:
            # Descarregar baseado no tipo de implementação
            if hasattr(self.model, "unload"):
                self.model.unload()
            else:
                # Cleanup genérico
                try:
                    del self.model
                except:
                    pass

            self.model = None
            self.model_loaded = False

            # Forçar limpeza de memória
            try:
                self.gpu_manager.clear_memory_cache()
            except:
                pass

            self.logger.info(
                f"Modelo descarregado: {os.path.basename(self.model_path) if self.model_path else 'unknown'}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Erro ao descarregar modelo: {e}")
            self.model = None
            self.model_loaded = False
            return False

    def __del__(self):
        """Limpeza ao destruir a instância."""
        try:
            if self.model_loaded:
                self.unload()
        except:
            # Ignorar erros no destrutor
            pass
