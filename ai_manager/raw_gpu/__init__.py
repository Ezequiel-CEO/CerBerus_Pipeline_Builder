# -*- coding: utf-8 -*-
"""
Módulo para aceleração GPU/CPU com implementação nativa do CerBerusFMK
Funciona como um orquestrador local de modelos com detecção automática de hardware
"""

import os
import sys
import logging
import importlib.util
import subprocess
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

# Configurar logger
logger = logging.getLogger("raw_gpu")

# Verificar variáveis de ambiente
DISABLE_CUDA = os.environ.get("DISABLE_CUDA", "0").lower() in ("1", "true", "yes")
FORCE_CPU = os.environ.get("FORCE_CPU", "0").lower() in ("1", "true", "yes")
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
CERBERUS_SIMULATED_MODE = os.environ.get("CERBERUS_SIMULATED_MODE", "0").lower() in (
    "1",
    "true",
    "yes",
)

# Flag global para disponibilidade de CUDA
CUDA_AVAILABLE = False
GPU_AVAILABLE = False

# Diretório base para modelos
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
os.makedirs(MODELS_DIR, exist_ok=True)


def detect_gpu_architecture() -> Optional[int]:
    """Detecta a arquitetura da GPU NVIDIA disponível."""
    try:
        # Tentar via subprocess
        import subprocess

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode == 0 and result.stdout:
                gpu_name = result.stdout.strip().lower()

                # Mapear nomes de GPU para arquiteturas
                gpu_to_arch = {
                    "1050": 61,  # Pascal (GTX 1050/1050 Ti)
                    "1060": 61,  # Pascal
                    "1070": 61,  # Pascal
                    "1080": 61,  # Pascal
                    "2060": 75,  # Turing
                    "2070": 75,  # Turing
                    "2080": 75,  # Turing
                    "3060": 86,  # Ampere
                    "3070": 86,  # Ampere
                    "3080": 86,  # Ampere
                    "3090": 86,  # Ampere
                    "4060": 89,  # Ada Lovelace
                    "4070": 89,  # Ada Lovelace
                    "4080": 89,  # Ada Lovelace
                    "4090": 89,  # Ada Lovelace
                }

                # Verificar qual GPU corresponde ao nome detectado
                for gpu_model, arch in gpu_to_arch.items():
                    if gpu_model in gpu_name:
                        logger.info(f"GPU detectada: {gpu_name} (Arquitetura: {arch})")
                        return arch

                logger.warning(
                    f"GPU detectada: {gpu_name}, mas arquitetura não mapeada. Usando padrão."
                )
            return 61  # Valor padrão conservador para GPUs não identificadas
        except Exception as e:
            logger.warning(f"Não foi possível detectar arquitetura GPU: {e}")
            return None
    except Exception as e:
        logger.warning(f"Erro ao detectar arquitetura GPU: {e}")
        return None


def check_cuda_availability() -> bool:
    """Verifica se CUDA está disponível nativamente no sistema."""
    if DISABLE_CUDA or FORCE_CPU:
        return False

    if CERBERUS_SIMULATED_MODE:
        logger.info("Modo simulado ativado. Ignorando verificação de CUDA/GPU.")
        return False

    try:
        # Verificar se CUDA está disponível usando comandos do sistema
        import subprocess

        try:
            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                global CUDA_AVAILABLE, GPU_AVAILABLE
                CUDA_AVAILABLE = True
                GPU_AVAILABLE = True
                logger.info("CUDA está disponível no sistema")
                return True
        except:
            pass

        # Verificar presença de bibliotecas CUDA
        cuda_paths = [
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/lib/x86_64-linux-gnu/libcudart.so",
            "/usr/lib/libcudart.so",
        ]

        # Declara as variáveis globais uma única vez para todo o bloco
        global CUDA_AVAILABLE, GPU_AVAILABLE

        for path in cuda_paths:
            if os.path.exists(path):
                CUDA_AVAILABLE = True
                GPU_AVAILABLE = True
                logger.info(f"Biblioteca CUDA encontrada em: {path}")
                return True

        logger.info("CUDA não está disponível no sistema")
        return False
    except Exception as e:
        logger.warning(f"Erro ao verificar disponibilidade de CUDA: {e}")
        return False


class ModelInfo:
    """Classe para armazenar informações sobre um modelo carregável"""

    def __init__(self, model_path: str, model_type: str, config: Dict[str, Any] = None):
        self.model_path = model_path
        self.model_type = model_type  # 'gguf', 'ggml', 'safetensors', etc.
        self.config = config or {}
        self.name = os.path.basename(os.path.dirname(model_path))

    def __str__(self) -> str:
        return f"ModelInfo(name={self.name}, type={self.model_type}, path={self.model_path})"


class ModelManager:
    """Gerenciador de modelos para o CerBerusFMK"""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.available_models = self._scan_models()

    def _scan_models(self) -> Dict[str, ModelInfo]:
        """Escaneia o diretório de modelos e identifica modelos disponíveis"""
        models = {}

        if not os.path.exists(self.models_dir):
            logger.warning(f"Diretório de modelos não existe: {self.models_dir}")
            return models

        # Procurar por pastas de modelo
        for model_folder in os.listdir(self.models_dir):
            folder_path = os.path.join(self.models_dir, model_folder)
            if not os.path.isdir(folder_path):
                continue

            # Verificar manifest.yaml ou config.json
            config_file = None
            if os.path.exists(os.path.join(folder_path, "manifest.yaml")):
                config_file = os.path.join(folder_path, "manifest.yaml")
            elif os.path.exists(os.path.join(folder_path, "config.json")):
                config_file = os.path.join(folder_path, "config.json")

            config = {}
            if config_file:
                try:
                    if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                        import yaml

                        with open(config_file, "r") as f:
                            config = yaml.safe_load(f)
                    elif config_file.endswith(".json"):
                        import json

                        with open(config_file, "r") as f:
                            config = json.load(f)
                except Exception as e:
                    logger.warning(
                        f"Erro ao carregar configuração do modelo {model_folder}: {e}"
                    )

            # Procurar por arquivos de modelo
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if not os.path.isfile(file_path):
                    continue

                # Identificar tipo de modelo
                model_type = None
                if file.endswith(".gguf"):
                    model_type = "gguf"
                elif file.endswith(".ggml"):
                    model_type = "ggml"
                elif file.endswith(".bin") and os.path.exists(
                    os.path.join(folder_path, "config.json")
                ):
                    model_type = "transformers"
                elif file.endswith(".safetensors"):
                    model_type = "safetensors"

                if model_type:
                    model_info = ModelInfo(file_path, model_type, config)
                    models[model_folder] = model_info
                    logger.info(f"Modelo encontrado: {model_folder} ({model_type})")
                    break

        return models

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Obtém informações sobre um modelo específico"""
        return self.available_models.get(model_name)

    def list_available_models(self) -> List[str]:
        """Lista os nomes dos modelos disponíveis"""
        return list(self.available_models.keys())


# Importar e inicializar o integrador de modelos
def _get_model_integrator():
    """Obtém o integrador de modelos nativos"""
    try:
        from .model_integrator import NativeModelIntegrator

        return NativeModelIntegrator()
    except ImportError:
        logger.warning(
            "Módulo de integração nativa não encontrado. Algumas funcionalidades estarão limitadas."
        )
        return None


# Singleton para o integrador de modelos
_MODEL_INTEGRATOR = _get_model_integrator()


class CerberusEngine:
    """
    Engine nativa do CerBerusFMK para inferência de modelos
    Fornece uma interface unificada para diferentes executores
    """

    def __init__(
        self,
        model_path_or_name: str,
        use_gpu: bool = True,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        **kwargs,
    ):
        # Determinar se é um caminho ou nome de modelo
        self.model_manager = ModelManager()

        if os.path.exists(model_path_or_name):
            # É um caminho direto para um arquivo
            model_dir = os.path.dirname(model_path_or_name)
            model_file = os.path.basename(model_path_or_name)
            model_type = self._detect_model_type(model_file)
            self.model_info = ModelInfo(model_path_or_name, model_type)
        else:
            # É um nome de modelo no diretório de modelos
            self.model_info = self.model_manager.get_model_info(model_path_or_name)

        if not self.model_info:
            logger.error(f"Modelo não encontrado: {model_path_or_name}")
            self._model_loaded = False
            return

        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.config = kwargs
        self.model = None
        self._initialize()

    def _detect_model_type(self, file_name: str) -> str:
        """Detecta o tipo de modelo com base no nome do arquivo"""
        if file_name.endswith(".gguf"):
            return "gguf"
        elif file_name.endswith(".ggml"):
            return "ggml"
        elif file_name.endswith(".bin"):
            return "transformers"
        elif file_name.endswith(".safetensors"):
            return "safetensors"
        else:
            return "unknown"

    def _initialize(self):
        """Inicializa o modelo usando implementação nativa"""
        global _MODEL_INTEGRATOR

        try:
            if _MODEL_INTEGRATOR is None:
                raise ImportError("Integrador de modelos nativos não disponível")

            # Carregar modelo usando o integrador
            self.model = _MODEL_INTEGRATOR.load_model(
                model_path=self.model_info.model_path,
                model_type=self.model_info.model_type,
                use_gpu=self.use_gpu,
            )

            if self.model is None:
                raise ValueError(
                    f"Falha ao carregar modelo: {self.model_info.model_path}"
                )

            self._model_loaded = True
            gpu_arch = detect_gpu_architecture() if self.use_gpu else None
            logger.info(
                f"Modelo carregado com sucesso. Arquitetura GPU: {gpu_arch if gpu_arch else 'N/A'}"
            )
        except Exception as e:
            logger.error(f"Falha ao inicializar modelo: {e}")
            self._model_loaded = False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        """Gera texto a partir do prompt usando o modelo escolhido"""
        if not self._model_loaded or self.model is None:
            return f"[ERRO] Modelo não está inicializado corretamente"

        if CERBERUS_SIMULATED_MODE:
            return f"[SIMULADO] Resposta do modelo {self.model_info.name} para: {prompt[:50]}..."

        try:
            # Usar o método nativo do modelo
            return self.model.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Erro durante geração: {e}")
            return f"[ERRO] Falha na geração: {str(e)}"

    def __del__(self):
        """Libera recursos alocados"""
        if hasattr(self, "model") and self.model is not None:
            del self.model


def create_optimized_engine(
    model_path: str,
    use_gpu: bool = True,
    n_gpu_layers: int = -1,
    n_ctx: int = 2048,
    **kwargs,
) -> Any:
    """
    Cria uma engine otimizada nativa do CerBerusFMK para inferência.

    Args:
        model_path: Caminho para o modelo ou nome do modelo na pasta models/
        use_gpu: Se deve usar GPU se disponível
        n_gpu_layers: Número de camadas para offload para GPU (-1 = todas)
        n_ctx: Tamanho do contexto
        **kwargs: Argumentos adicionais para o modelo

    Returns:
        Engine nativa do CerBerusFMK para inferência
    """
    # Verificar variáveis de ambiente
    if DISABLE_CUDA or FORCE_CPU:
        logger.info("GPU desativada por variável de ambiente")
        use_gpu = False

    # Se GPU foi solicitada, verificar disponibilidade
    if use_gpu:
        cuda_available = check_cuda_availability()
        if not cuda_available:
            logger.warning("CUDA não disponível. Usando CPU.")
            use_gpu = False

    # Criar engine do CerBerusFMK
    engine = CerberusEngine(
        model_path_or_name=model_path,
        use_gpu=use_gpu,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        **kwargs,
    )

    if not engine._model_loaded:
        logger.error(f"Falha ao criar engine para modelo: {model_path}")
        return None

    return engine


def list_available_models() -> List[str]:
    """Lista todos os modelos disponíveis no diretório padrão"""
    manager = ModelManager()
    return manager.list_available_models()


def create_model_template(model_name: str, model_type: str = "gguf") -> str:
    """
    Cria uma estrutura de pasta e arquivos de configuração para um novo modelo.

    Args:
        model_name: Nome para a pasta do modelo
        model_type: Tipo de modelo (gguf, ggml, transformers, safetensors)

    Returns:
        Caminho para a pasta criada
    """
    model_dir = os.path.join(MODELS_DIR, model_name)

    if os.path.exists(model_dir):
        logger.warning(f"Pasta de modelo já existe: {model_dir}")
        return model_dir

    # Criar pasta do modelo
    os.makedirs(model_dir, exist_ok=True)

    # Criar arquivo manifest.yaml com configuração básica
    manifest_content = f"""
# Configuração do modelo {model_name}
model_type: {model_type}
description: "Modelo {model_type.upper()} para CerBerusFMK"
parameters:
  n_ctx: 2048
  n_gpu_layers: -1  # -1 = todas as camadas
  use_gpu: true
  temperature: 0.8
  top_p: 0.95
  max_tokens: 256
"""

    with open(os.path.join(model_dir, "manifest.yaml"), "w") as f:
        f.write(manifest_content)

    # Criar arquivo README com instruções
    readme_content = f"""# Modelo {model_name}

Este é um modelo do tipo {model_type} para o CerBerusFMK.

## Como usar

1. Coloque o arquivo do modelo ({model_type}) nesta pasta
2. Ajuste os parâmetros no arquivo manifest.yaml conforme necessário
3. O CerBerusFMK detectará automaticamente o modelo

## Parâmetros recomendados

* Para GPU de 4GB: `n_gpu_layers=20`
* Para GPU de 8GB: `n_gpu_layers=32`
* Para GPU de 12GB+: `n_gpu_layers=-1` (todas as camadas)
"""

    with open(os.path.join(model_dir, "README.md"), "w") as f:
        f.write(readme_content)

    logger.info(f"Template de modelo criado com sucesso: {model_dir}")
    return model_dir


# Verificar CUDA ao carregar o módulo
CUDA_AVAILABLE = check_cuda_availability()
