import os
import gc
import logging
from typing import Dict, Any, Optional, List

from llama_cpp import Llama
from .base_loader import BaseModelLoader

# Configurar logger local
logger = logging.getLogger(__name__)


class LlamaCppLoader(BaseModelLoader):
    """
    Implementação de alta performance para carregar modelos GGUF com llama-cpp-python
    """

    def __init__(self, model_path: str, **kwargs):
        """
        Inicializa o loader LlamaCpp.

        Args:
            model_path: Caminho para o modelo GGUF
            **kwargs: Argumentos adicionais para o LlamaCpp
        """
        super().__init__(model_path)
        self.kwargs = kwargs
        self.llm = None

        logger.info(f"LlamaCppLoader inicializado: {os.path.basename(model_path)}")

    def load(self) -> None:
        """
        Carrega o modelo GGUF usando llama.cpp para máxima performance.
        """
        try:
            logger.info(f"Carregando modelo GGUF: {self.model_path}")
            self.llm = Llama(model_path=self.model_path, **self.kwargs)
            logger.info("Modelo GGUF carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo GGUF: {str(e)}")
            raise

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Gera texto a partir de um prompt usando o modelo GGUF.

        Args:
            prompt: Texto de entrada
            **kwargs: Argumentos adicionais para geração

        Returns:
            str: Texto gerado
        """
        if not self.llm:
            raise RuntimeError("Modelo não carregado. Chame load() primeiro.")

        try:
            output = self.llm(prompt, **kwargs)
            return output["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Erro ao gerar texto: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo.

        Returns:
            Dicionário com informações do modelo
        """
        if self.llm is None:
            return {"status": "não carregado"}

        return {
            "model_path": self.model_path,
            "model_name": os.path.basename(self.model_path),
            "n_gpu_layers": self.kwargs.get("n_gpu_layers", -1),
            "context_size": self.kwargs.get("n_ctx", 2048),
            "batch_size": self.kwargs.get("n_batch", 512),
        }

    def unload(self) -> None:
        """
        Descarrega o modelo da memória.
        """
        self.llm = None
        logger.info("Modelo GGUF descarregado da memória")
        gc.collect()
