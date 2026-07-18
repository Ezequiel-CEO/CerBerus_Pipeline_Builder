import os
import logging
from typing import Dict, Any, Optional, List, Union

from cerberus_api.pipeline_builder.ai_manager.base_manager import BaseModelLoader
from cerberus_api.pipeline_builder.ai_manager.eter.huggingface_loader import (
    HuggingFaceLoader,
)
from cerberus_api.pipeline_builder.ai_manager.eter.llamacpp_loader import LlamaCppLoader
from modules.eter.adaptadores.utils.gguf_loader import GGUFLoader

logger = logging.getLogger(__name__)


class ModelLoaderFactory:
    """
    Fábrica para criar carregadores de modelos específicos com base no tipo de modelo.
    """

    @staticmethod
    def criar_carregador(
        caminho_modelo: str, tipo_modelo: Optional[str] = None, **kwargs
    ) -> Optional[BaseModelLoader]:
        """
        Cria um carregador de modelo adequado com base no tipo ou extensão do modelo.

        Args:
            caminho_modelo: Caminho para o arquivo do modelo
            tipo_modelo: Tipo explícito do modelo (opcional)
            **kwargs: Argumentos adicionais para o carregador

        Returns:
            Um carregador de modelo apropriado ou None se não for possível determinar
        """
        logger.info(f"Criando carregador para modelo: {caminho_modelo}")

        # Se o tipo for explicitamente fornecido
        if tipo_modelo:
            return ModelLoaderFactory._criar_por_tipo(
                caminho_modelo, tipo_modelo, **kwargs
            )

        # Tenta inferir o tipo a partir do caminho
        extensao = os.path.splitext(caminho_modelo)[1].lower()
        nome_arquivo = os.path.basename(caminho_modelo).lower()

        # Verifica se é um modelo GGUF
        if extensao == ".gguf" or "gguf" in nome_arquivo:
            logger.info(f"Modelo GGUF detectado: {caminho_modelo}")
            return GGUFLoader(caminho_modelo, **kwargs)

        # Verifica se é um modelo Hugging Face (pasta ou arquivo .bin)
        if os.path.isdir(caminho_modelo) or extensao == ".bin":
            logger.info(f"Modelo Hugging Face detectado: {caminho_modelo}")
            return HuggingFaceLoader(caminho_modelo, **kwargs)

        # Verifica se é um arquivo GGML para llama.cpp
        if "ggml" in nome_arquivo or "llama" in nome_arquivo:
            logger.info(f"Modelo GGML/LlamaCPP detectado: {caminho_modelo}")
            return LlamaCppLoader(caminho_modelo, **kwargs)

        # Tenta detectar pelo conteúdo da primeira linha do arquivo
        # (implementação adicional se necessário)

        logger.warning(
            f"Não foi possível determinar o tipo de modelo para: {caminho_modelo}"
        )
        return None

    @staticmethod
    def _criar_por_tipo(
        caminho_modelo: str, tipo_modelo: str, **kwargs
    ) -> Optional[BaseModelLoader]:
        """
        Cria um carregador com base no tipo explícito de modelo.

        Args:
            caminho_modelo: Caminho para o arquivo do modelo
            tipo_modelo: Tipo de modelo (huggingface, llamacpp, etc.)
            **kwargs: Argumentos adicionais para o carregador

        Returns:
            Um carregador de modelo apropriado ou None se o tipo for desconhecido
        """
        tipo_modelo = tipo_modelo.lower()

        if tipo_modelo in ["huggingface", "hf", "transformers"]:
            return HuggingFaceLoader(caminho_modelo, **kwargs)

        if tipo_modelo in ["llamacpp", "llama", "ggml"]:
            return LlamaCppLoader(caminho_modelo, **kwargs)

        if tipo_modelo in ["gguf"]:
            return GGUFLoader(caminho_modelo, **kwargs)

        # Adicione mais tipos conforme necessário

        logger.warning(f"Tipo de modelo desconhecido: {tipo_modelo}")
        return None
