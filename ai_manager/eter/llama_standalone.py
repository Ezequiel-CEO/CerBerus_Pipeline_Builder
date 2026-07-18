#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo LlamaCppStandalone - Implementação standalone para modelos GGUF
Solução independente que não depende de outras partes do sistema.
"""

import os
import sys
import gc
import time
import logging
from typing import Dict, Any, Optional, List

# Configurar logger simples
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Verificar se llama-cpp-python está disponível
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
    logger.info("llama-cpp-python está disponível")
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python não encontrado. Funcionalidade GGUF desativada.")
    Llama = None


class LlamaCppStandalone:
    """
    Implementação standalone para modelos GGUF usando llama-cpp-python.
    Esta classe é independente e não requer outras partes do sistema.
    """

    def __init__(self, model_path: str, **kwargs):
        """
        Inicializa o loader.

        Args:
            model_path: Caminho para o modelo GGUF
            **kwargs: Configurações adicionais para o modelo
        """
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python não está disponível. Esta funcionalidade está desativada.")
            raise ImportError("llama-cpp-python não encontrado")
            
        self.model_path = model_path
        self.llm = None
        self.kwargs = kwargs

        # Verificar se o arquivo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")

        # Definir valores padrão se não fornecidos
        if "n_gpu_layers" not in self.kwargs:
            self.kwargs["n_gpu_layers"] = -1
        if "n_ctx" not in self.kwargs:
            self.kwargs["n_ctx"] = 2048
        if "n_batch" not in self.kwargs:
            self.kwargs["n_batch"] = 512

        logger.info(
            f"LlamaCppStandalone inicializado para: {os.path.basename(model_path)}"
        )

    def load(self) -> bool:
        """
        Carrega o modelo GGUF.

        Returns:
            bool: True se o modelo foi carregado com sucesso, False caso contrário
        """
        try:
            logger.info(f"Carregando modelo GGUF: {self.model_path}")
            start_time = time.time()

            self.llm = Llama(model_path=self.model_path, **self.kwargs)

            load_time = time.time() - start_time
            logger.info(
                f"Modelo GGUF carregado com sucesso em {load_time:.2f} segundos"
            )
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo GGUF: {str(e)}")
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Gera texto a partir de um prompt.

        Args:
            prompt: Texto para gerar a partir
            **kwargs: Configurações de geração

        Returns:
            str: Texto gerado pelo modelo
        """
        if not self.llm:
            logger.warning("Modelo não carregado. Carregando...")
            if not self.load():
                return "Erro: Falha ao carregar modelo"

        try:
            logger.info(f"Gerando texto para prompt: {prompt[:50]}...")
            start_time = time.time()

            # Configurações padrão
            gen_params = {
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "repeat_penalty": 1.1,
            }

            # Atualizar com kwargs fornecidos
            gen_params.update(kwargs)

            # Gerar resposta
            output = self.llm(prompt, **gen_params)

            gen_time = time.time() - start_time
            logger.info(f"Texto gerado em {gen_time:.2f} segundos")

            return output["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Erro ao gerar texto: {str(e)}")
            return f"Erro: {str(e)}"

    def get_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo.

        Returns:
            Dict[str, Any]: Informações do modelo
        """
        info = {
            "model_path": self.model_path,
            "model_name": os.path.basename(self.model_path),
            "status": "carregado" if self.llm else "não carregado",
            "n_gpu_layers": self.kwargs.get("n_gpu_layers", -1),
            "n_ctx": self.kwargs.get("n_ctx", 2048),
            "n_batch": self.kwargs.get("n_batch", 512),
        }

        # Tentar obter metadados do modelo
        if self.llm:
            try:
                info["metadata"] = self.llm.model_metadata()
            except:
                info["metadata"] = "Não disponível"

        return info

    def unload(self) -> None:
        """
        Descarrega o modelo da memória.
        """
        self.llm = None
        logger.info("Modelo GGUF descarregado da memória")
        gc.collect()


# Função de ajuda para testar o módulo diretamente
def test_modelo(model_path: str, prompt: str = None):
    """
    Função de teste para o módulo.

    Args:
        model_path: Caminho para o modelo GGUF
        prompt: Prompt opcional para testar
    """
    if not os.path.exists(model_path):
        logger.error(f"Modelo não encontrado: {model_path}")
        return

    logger.info(
        f"Testando LlamaCppStandalone com modelo: {os.path.basename(model_path)}"
    )

    # Criar instância
    loader = LlamaCppStandalone(
        model_path=model_path, n_gpu_layers=-1, n_ctx=2048, n_batch=512
    )

    # Carregar modelo
    if not loader.load():
        logger.error("Falha ao carregar modelo")
        return

    # Obter informações
    info = loader.get_info()
    logger.info("Informações do modelo:")
    for key, value in info.items():
        if key != "metadata":
            logger.info(f"  {key}: {value}")

    # Testar geração se prompt fornecido
    if prompt:
        logger.info(f"Testando geração com prompt: {prompt}")
        resposta = loader.generate(prompt=prompt, max_tokens=200)
        logger.info(f"Resposta:\n{resposta}")

    # Descarregar
    loader.unload()
    logger.info("Teste concluído")


# Permitir execução direta do módulo para teste
if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        prompt = (
            sys.argv[2]
            if len(sys.argv) > 2
            else "Explique o que é inteligência artificial em português:"
        )
        test_modelo(model_path, prompt)
    else:
        logger.error("Uso: python llama_standalone.py <caminho_modelo> [prompt]")
        sys.exit(1)
