import logging
from typing import Dict, Any, Optional

# Importação dos loaders
from .phi_pro import PhiPro
from .base_loader import BaseModelLoader
from .transformers_loader import TransformersLoader
from .llama_cpp_loader import LlamaCppLoader
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """
    Factory para criação de modelos de IA para o framework.
    Implementação profissional e limpa.
    """

    @staticmethod
    def create_model(model_type: str, **kwargs) -> Any:
        """
        Cria um modelo baseado no tipo especificado.

        Args:
            model_type: Tipo de modelo ('phi-2', etc.)
            **kwargs: Parâmetros específicos para o modelo

        Returns:
            Instância do modelo solicitado
        """
        logger.info(f"Criando modelo: {model_type}")

        if model_type.lower() in ["phi", "phi-2", "phi2"]:
            # Verificar parâmetros específicos para Phi-2
            model_id = kwargs.get("model_id", "microsoft/phi-2")
            return PhiPro(model_id=model_id)

        # Outros modelos podem ser adicionados aqui no futuro

        # Fallback para modelo não suportado
        raise ValueError(f"Modelo não suportado: {model_type}")

    @staticmethod
    def list_available_models() -> Dict[str, Dict[str, Any]]:
        """
        Lista todos os modelos disponíveis através desta factory.

        Returns:
            Dicionário com informações sobre os modelos disponíveis
        """
        return {
            "phi-2": {
                "description": "Modelo leve e poderoso da Microsoft",
                "size": "2.7B parâmetros",
                "requirements": {"gpu_memory": "3GB (mínimo)", "cuda": "Recomendado"},
                "capabilities": [
                    "Geração de texto",
                    "Codificação",
                    "Suporte a português",
                ],
            }
            # Outros modelos podem ser adicionados aqui
        }

    @staticmethod
    def create_loader(model_type: str, model_path: str, **kwargs) -> BaseModelLoader:
        """
        Cria um loader apropriado para o tipo de modelo especificado.

        Args:
            model_type: Tipo do modelo ('transformers', 'llama_cpp', etc)
            model_path: Caminho para o modelo
            **kwargs: Argumentos adicionais para o loader

        Returns:
            BaseModelLoader: Instância do loader apropriado
        """
        try:
            if model_type == "transformers":
                return TransformersLoader(model_path, **kwargs)
            elif model_type == "llama_cpp":
                return LlamaCppLoader(model_path, **kwargs)
            else:
                raise ValueError(f"Tipo de modelo não suportado: {model_type}")
        except Exception as e:
            logger.error(f"Erro ao criar loader: {str(e)}")
            raise
