# -*- coding: utf-8 -*-
"""
Integrador de modelos nativos com o gerenciador raw_gpu
"""

import os
import logging
import importlib
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

logger = logging.getLogger("model_integrator")


def load_native_module(module_path: str) -> Optional[Any]:
    """
    Carrega dinamicamente um módulo nativo.

    Args:
        module_path: Caminho do módulo (ex: 'cerberus_api.pipeline_builder.ai_manager.native_loaders.gguf_loader')

    Returns:
        Módulo carregado ou None se falhar
    """
    try:
        module = importlib.import_module(module_path)
        logger.info(f"Módulo nativo carregado com sucesso: {module_path}")
        return module
    except ImportError as e:
        logger.error(f"Erro ao carregar módulo nativo {module_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Erro inesperado ao carregar módulo {module_path}: {e}")
        return None


class NativeModelIntegrator:
    """
    Integra loaders nativos com o sistema raw_gpu
    Fornece uma interface uniforme para diferentes tipos de modelos
    """

    def __init__(self):
        self.loaders = {}
        self.tokenizers = {}
        self.inference_engines = {}
        self._load_native_modules()

    def _load_native_modules(self):
        """Carrega todos os módulos nativos disponíveis"""
        # Carregar módulo de tokenização
        tokenizer_module = load_native_module(
            "cerberus_api.pipeline_builder.ai_manager.native_loaders.tokenizer"
        )
        if tokenizer_module:
            self.tokenizers["bpe"] = tokenizer_module.BPETokenizer
            self.tokenizers["simple"] = tokenizer_module.SimpleTokenizer
            self.tokenizers["create"] = tokenizer_module.create_tokenizer

        # Carregar módulo de inferência
        inference_module = load_native_module(
            "cerberus_api.pipeline_builder.ai_manager.native_loaders.inference"
        )
        if inference_module:
            self.inference_engines["native"] = inference_module.NativeInference
            self.inference_engines["create"] = inference_module.create_inference_engine

        # Carregar loader GGUF
        gguf_module = load_native_module(
            "cerberus_api.pipeline_builder.ai_manager.native_loaders.gguf_loader"
        )
        if gguf_module:
            self.loaders["gguf"] = gguf_module.GGUFModel
            self.loaders["gguf_reader"] = gguf_module.GGUFReader

        # Adicionar outros loaders aqui à medida que forem implementados

    def create_tokenizer(
        self,
        tokenizer_type: str,
        vocab_path: Optional[str] = None,
        merges_path: Optional[str] = None,
    ):
        """
        Cria um tokenizador.

        Args:
            tokenizer_type: Tipo de tokenizador (bpe, simple)
            vocab_path: Caminho para o arquivo de vocabulário
            merges_path: Caminho para o arquivo de mesclagens (BPE)

        Returns:
            Instância do tokenizador
        """
        if "create" not in self.tokenizers:
            logger.error("Módulo de tokenização não disponível")
            return None

        return self.tokenizers["create"](tokenizer_type, vocab_path, merges_path)

    def load_model(
        self, model_path: str, model_type: str = None, use_gpu: bool = True
    ) -> Optional[Any]:
        """
        Carrega um modelo usando o loader nativo apropriado.

        Args:
            model_path: Caminho para o arquivo do modelo
            model_type: Tipo do modelo (gguf, ggml, etc.). Se None, será inferido da extensão
            use_gpu: Se deve usar aceleração GPU

        Returns:
            Modelo carregado ou None se falhar
        """
        if not os.path.exists(model_path):
            logger.error(f"Arquivo de modelo não encontrado: {model_path}")
            return None

        # Inferir tipo de modelo a partir da extensão se não for especificado
        if model_type is None:
            ext = os.path.splitext(model_path)[1].lower()
            if ext == ".gguf":
                model_type = "gguf"
            elif ext == ".ggml":
                model_type = "ggml"
            elif ext == ".bin":
                model_type = "transformers"
            elif ext == ".safetensors":
                model_type = "safetensors"
            else:
                logger.error(
                    f"Não foi possível inferir o tipo do modelo a partir da extensão: {ext}"
                )
                return None

        # Verificar se temos loader para este tipo
        if model_type not in self.loaders:
            logger.error(f"Loader não disponível para o tipo de modelo: {model_type}")
            return None

        try:
            # Carregar modelo usando loader apropriado
            model = self.loaders[model_type](model_path, use_gpu=use_gpu)
            logger.info(f"Modelo {model_type} carregado com sucesso: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_path}: {e}")
            return None

    def create_inference_engine(
        self, weights: Dict[str, np.ndarray], config: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Cria um motor de inferência nativo.

        Args:
            weights: Dicionário de tensores do modelo
            config: Configuração do modelo

        Returns:
            Motor de inferência
        """
        if "create" not in self.inference_engines:
            logger.error("Módulo de inferência não disponível")
            return None

        try:
            engine = self.inference_engines["create"](weights, config)
            return engine
        except Exception as e:
            logger.error(f"Erro ao criar motor de inferência: {e}")
            return None
