from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModelLoader(ABC):
    """Classe base abstrata para loaders de modelos."""

    def __init__(self, model_path: str, **kwargs):
        """
        Inicializa o loader.

        Args:
            model_path: Caminho para o modelo
            **kwargs: Argumentos adicionais específicos do loader
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self) -> None:
        """Carrega o modelo e o tokenizer."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Gera texto a partir de um prompt.

        Args:
            prompt: Texto de entrada
            **kwargs: Argumentos adicionais para geração

        Returns:
            str: Texto gerado
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Descarrega o modelo da memória."""
        pass
