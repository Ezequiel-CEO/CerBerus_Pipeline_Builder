#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GGUFManager: Gerenciador de modelos GGUF para o CerBerusFMK.

Este módulo implementa um gerenciador para modelos GGUF (GPT-Generated Unified Format),
sem depender de bibliotecas de terceiros como llama-cpp, utilizando apenas a
implementação nativa do CerBerusFMK.
"""

import os
import time
import logging
import json
from typing import Dict, List, Any, Optional, Union, Generator, Tuple

import torch
import numpy as np

from cerberus_api.utils.logging_config import get_logger
from cerberus_api.pipeline_builder.utils.memory_manager import MemoryManager

# Verificar disponibilidade da implementação nativa
NATIVE_GGUF_AVAILABLE = False
try:
    # Tentar importar as classes nativas do CerBerusFMK para processamento GGUF
    from libs.CerBerusFMK.CerBerusFMK.llm_local.native_gguf import NativeGGUFRunner
    from libs.CerBerusFMK.CerBerusFMK.llm_local.gguf_model import GGUFReader
    from libs.CerBerusFMK.CerBerusFMK.inference_engine.engine import InferenceEngine

    NATIVE_GGUF_AVAILABLE = True
except ImportError as e:
    print(f"Implementação nativa GGUF não disponível: {e}")
    print("Será usada implementação simulada")

# Configuração de logging
logger = get_logger("gguf_manager")


class NativeModel:
    """
    Implementação nativa de modelo GGUF usando componentes do CerBerusFMK.

    Esta classe funciona como uma ponte entre o GGUFManager e a implementação
    do CerBerusFMK, sem depender de bibliotecas externas.
    """

    def __init__(
        self,
        model_path: str,
        context_size: int = 4096,
        use_gpu: bool = True,
        verbose: bool = False,
    ):
        """
        Inicializa o modelo GGUF nativo.

        Args:
            model_path: Caminho para o arquivo de modelo GGUF
            context_size: Tamanho do contexto (janela de tokens)
            use_gpu: Se deve usar GPU quando disponível
            verbose: Modo detalhado para debug
        """
        self.logger = logger
        self.model_path = os.path.abspath(model_path)
        self.context_size = context_size
        self.use_gpu = use_gpu
        self.verbose = verbose

        # Estado do modelo
        self.loaded = False
        self.model = None

        # Verificar se o modelo existe
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Arquivo de modelo não encontrado: {self.model_path}"
            )

        # Informações sobre o modelo
        self.model_info = self._get_model_info()

        if NATIVE_GGUF_AVAILABLE:
            self.logger.info(
                f"NativeModel inicializado: {os.path.basename(model_path)}"
            )
        else:
            self.logger.warning(
                f"NativeModel em modo simulado: {os.path.basename(model_path)}"
            )

    def _get_model_info(self) -> Dict[str, Any]:
        """Obtém informações básicas sobre o modelo."""
        # Tamanho do arquivo em GB
        model_size_bytes = os.path.getsize(self.model_path)
        model_size_gb = model_size_bytes / (1024 * 1024 * 1024)

        # Nome base do modelo (sem diretório e extensão)
        model_name = os.path.basename(self.model_path)
        model_name = os.path.splitext(model_name)[0]

        # Configuração padrão
        model_info = {
            "name": model_name,
            "path": self.model_path,
            "size_bytes": model_size_bytes,
            "size_gb": model_size_gb,
            "n_ctx": self.context_size,
        }

        # Verificar se há um arquivo de configuração
        config_path = os.path.splitext(self.model_path)[0] + ".json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    model_info.update(config_data)
            except Exception as e:
                self.logger.warning(f"Erro ao carregar configuração: {e}")

        return model_info

    def load(self) -> bool:
        """
        Carrega o modelo em memória.

        Returns:
            True se carregado com sucesso
        """
        if self.loaded and self.model is not None:
            self.logger.debug(f"Modelo {self.model_info['name']} já está carregado")
            return True

        if not NATIVE_GGUF_AVAILABLE:
            self.logger.error("Implementação nativa GGUF não disponível")
            self._create_simulated_model()
            return True

        try:
            self.logger.info(f"Carregando modelo nativo: {self.model_path}")

            # Implementação usando o NativeGGUFRunner do CerBerusFMK
            self.model = NativeGGUFRunner(
                model_path=self.model_path,
                context_size=self.context_size,
                use_gpu=self.use_gpu,
                verbose=self.verbose,
            )

            self.loaded = True
            self.logger.info(f"Modelo {self.model_info['name']} carregado com sucesso")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo nativo: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

            # Criar modelo simulado em caso de erro
            self._create_simulated_model()
            return True

    def _create_simulated_model(self):
        """Cria um modelo simulado para testes e desenvolvimento."""
        self.model = type(
            "SimulatedModel",
            (),
            {
                "generate": lambda *args, **kwargs: {
                    "text": f"[Texto simulado] O modelo nativo GGUF não está disponível."
                }
            },
        )()
        self.loaded = True

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        streaming: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Gera texto a partir do prompt fornecido.

        Args:
            prompt: Texto de entrada
            max_tokens: Número máximo de tokens a gerar
            temperature: Temperatura para amostragem (0.0-1.0)
            top_p: Valor para amostragem núcleo (0.0-1.0)
            streaming: Se deve gerar em modo streaming
            **kwargs: Parâmetros adicionais

        Returns:
            Dicionário com o texto gerado ou gerador para streaming
        """
        if not self.loaded:
            success = self.load()
            if not success:
                return {"text": "Erro: não foi possível carregar o modelo"}

        try:
            # Montar configuração para geração
            generation_config = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "streaming": streaming,
            }
            generation_config.update(kwargs)

            # Gerar o texto
            result = self.model.generate(prompt, generation_config)

            # Se for streaming, retornar o gerador diretamente
            if streaming and hasattr(result, "__iter__"):
                return result

            # Para modo não-streaming, retornar o texto em um dicionário
            if isinstance(result, dict) and "text" in result:
                return result
            elif isinstance(result, str):
                return {"text": result}
            else:
                return {"text": str(result)}

        except Exception as e:
            self.logger.error(f"Erro na geração: {e}")
            return {"text": f"Erro: {str(e)}"}

    def unload(self) -> bool:
        """
        Descarrega o modelo da memória.

        Returns:
            True se descarregado com sucesso
        """
        if not self.loaded:
            return True

        try:
            # Verificar se existe método unload
            if hasattr(self.model, "unload"):
                self.model.unload()

            # Liberar referência ao modelo
            self.model = None
            self.loaded = False

            # Forçar limpeza de memória
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            import gc

            gc.collect()

            self.logger.info(f"Modelo {self.model_info['name']} descarregado")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao descarregar modelo: {e}")
            return False


class GGUFManager:
    """
    Gerenciador para múltiplos modelos GGUF.

    Permite carregar vários modelos e gerenciar uso de memória
    carregando/descarregando conforme necessário.
    """

    def __init__(
        self,
        models_dir: str = "~/.cerberusfmk/models",
        max_models_loaded: int = 1,
        memory_threshold: float = 0.8,
        auto_unload: bool = True,
    ):
        """
        Inicializa o gerenciador de modelos GGUF.

        Args:
            models_dir: Diretório base para modelos
            max_models_loaded: Número máximo de modelos carregados simultaneamente
            memory_threshold: Limiar de uso de memória para descarga automática
            auto_unload: Se True, descarrega modelos automaticamente
        """
        self.logger = logger
        self.models_dir = os.path.abspath(os.path.expanduser(models_dir))
        self.max_models_loaded = max_models_loaded
        self.memory_threshold = memory_threshold
        self.auto_unload = auto_unload

        # Gerenciador de memória
        self.memory_manager = MemoryManager()

        # Dicionário de modelos
        self.models: Dict[str, NativeModel] = {}

        # Criar diretório se não existir
        os.makedirs(self.models_dir, exist_ok=True)

        # Descobrir modelos disponíveis
        self._discover_models()

        self.logger.info(
            f"GGUFManager inicializado. Modelos disponíveis: {len(self.models)}"
        )

        # Lista de modelos carregados recentemente (para LRU)
        self.last_used: Dict[str, float] = {}

    def _discover_models(self) -> None:
        """Descobre modelos GGUF disponíveis no diretório."""
        gguf_files = []

        # Procurar arquivos GGUF
        for root, _, files in os.walk(self.models_dir):
            for file in files:
                if file.endswith((".gguf", ".bin")) and "gguf" in file.lower():
                    full_path = os.path.join(root, file)
                    gguf_files.append(full_path)

        # Registrar modelos encontrados
        for file_path in gguf_files:
            model_name = os.path.basename(file_path)
            model_name = os.path.splitext(model_name)[0]

            # Criar modelo mas não carregar ainda
            try:
                model = NativeModel(model_path=file_path, verbose=False)
                self.models[model_name] = model
                self.logger.info(f"Modelo registrado: {model_name}")
            except Exception as e:
                self.logger.error(f"Erro ao registrar modelo {model_name}: {e}")

    def add_model(
        self, model_path: str, load_immediately: bool = False
    ) -> Optional[str]:
        """
        Adiciona um novo modelo ao gerenciador.

        Args:
            model_path: Caminho para o arquivo de modelo GGUF
            load_immediately: Se True, carrega o modelo imediatamente

        Returns:
            Nome do modelo adicionado ou None se falhar
        """
        try:
            # Verificar se o modelo existe
            if not os.path.exists(model_path):
                self.logger.error(f"Arquivo de modelo não encontrado: {model_path}")
                return None

            # Obter nome do modelo
            model_name = os.path.basename(model_path)
            model_name = os.path.splitext(model_name)[0]

            # Verificar se já existe
            if model_name in self.models:
                self.logger.warning(f"Modelo {model_name} já existe. Substituindo.")

                # Descarregar o modelo antigo se estiver carregado
                if self.models[model_name].loaded:
                    self.models[model_name].unload()

            # Criar novo modelo
            model = NativeModel(model_path=model_path, verbose=False)

            # Adicionar ao dicionário
            self.models[model_name] = model

            # Carregar o modelo se solicitado
            if load_immediately:
                model.load()
                self.last_used[model_name] = time.time()

            self.logger.info(f"Modelo adicionado: {model_name}")
            return model_name

        except Exception as e:
            self.logger.error(f"Erro ao adicionar modelo: {e}")
            return None

    def get_model(self, model_name: str) -> Optional[NativeModel]:
        """
        Obtém um modelo pelo nome.

        Args:
            model_name: Nome do modelo

        Returns:
            Instância do modelo ou None se não encontrado
        """
        if model_name not in self.models:
            self.logger.warning(f"Modelo {model_name} não encontrado")
            return None

        # Checar se precisa descarregar outros modelos
        self._manage_memory()

        # Registrar último acesso
        self.last_used[model_name] = time.time()

        # Retornar o modelo
        return self.models[model_name]

    def load_model(self, model_name: str) -> bool:
        """
        Carrega um modelo específico.

        Args:
            model_name: Nome do modelo

        Returns:
            True se carregado com sucesso
        """
        if model_name not in self.models:
            self.logger.warning(f"Modelo {model_name} não encontrado")
            return False

        # Checar se precisa descarregar outros modelos
        self._manage_memory()

        # Carregar o modelo
        success = self.models[model_name].load()

        # Registrar último acesso
        if success:
            self.last_used[model_name] = time.time()

        return success

    def unload_model(self, model_name: str) -> bool:
        """
        Descarrega um modelo específico.

        Args:
            model_name: Nome do modelo

        Returns:
            True se descarregado com sucesso
        """
        if model_name not in self.models:
            return True  # Já não existe

        # Remover do registro de último acesso
        if model_name in self.last_used:
            del self.last_used[model_name]

        return self.models[model_name].unload()

    def remove_model(self, model_name: str) -> bool:
        """
        Remove um modelo do gerenciador.

        Args:
            model_name: Nome do modelo

        Returns:
            True se removido com sucesso
        """
        if model_name not in self.models:
            return True  # Já não existe

        # Descarregar primeiro
        self.models[model_name].unload()

        # Remover do dicionário
        del self.models[model_name]

        # Remover do registro de último acesso
        if model_name in self.last_used:
            del self.last_used[model_name]

        self.logger.info(f"Modelo removido: {model_name}")
        return True

    def _manage_memory(self) -> None:
        """
        Gerencia a memória, descarregando modelos quando necessário.
        """
        if not self.auto_unload:
            return

        # Contar modelos carregados
        loaded_models = [name for name, model in self.models.items() if model.loaded]

        # Verificar uso de memória
        memory_info = self.memory_manager.get_memory_usage()
        ram_usage = memory_info.get("ram_used_mb", 0)
        vram_usage = memory_info.get("vram_used_mb", 0)
        ram_total = memory_info.get("ram_total_mb", 1)
        vram_total = memory_info.get("vram_total_mb", 1)

        ram_percent = ram_usage / ram_total if ram_total > 0 else 0
        vram_percent = vram_usage / vram_total if vram_total > 0 else 0

        # Decidir se precisa descarregar modelos
        need_unload = (
            len(loaded_models) > self.max_models_loaded
            or ram_percent > self.memory_threshold
            or vram_percent > self.memory_threshold
        )

        if need_unload and loaded_models:
            self.logger.info(
                f"Gerenciando memória. Uso: RAM {ram_percent:.1%}, VRAM {vram_percent:.1%}"
            )

            # Ordenar por tempo de último uso (mais antigo primeiro)
            if self.last_used:
                loaded_models.sort(key=lambda m: self.last_used.get(m, 0))

                # Descarregar modelos até ficar abaixo dos limites
                for model_name in loaded_models[: -self.max_models_loaded]:
                    self.logger.info(
                        f"Descarregando modelo {model_name} para economizar memória"
                    )
                    self.models[model_name].unload()

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lista modelos disponíveis.

        Returns:
            Lista de informações sobre os modelos
        """
        models_info = []

        for name, model in self.models.items():
            info = {
                "name": name,
                "loaded": model.loaded,
                "path": model.model_path,
                "size_gb": model.model_info.get("size_gb", 0),
                "context_size": model.context_size,
            }
            models_info.append(info)

        return models_info

    def is_model_registered(self, model_name: str) -> bool:
        """
        Verifica se um modelo está registrado.

        Args:
            model_name: Nome do modelo

        Returns:
            True se o modelo estiver registrado
        """
        return model_name in self.models

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Obtém informações sobre uso de memória.

        Returns:
            Dicionário com informações de uso de memória
        """
        memory_info = self.memory_manager.get_memory_usage()

        return {
            "ram_mb": memory_info.get("ram_used_mb", 0),
            "ram_total_mb": memory_info.get("ram_total_mb", 0),
            "ram_percent": memory_info.get("ram_percent", 0),
            "vram_mb": memory_info.get("vram_used_mb", 0),
            "vram_total_mb": memory_info.get("vram_total_mb", 0),
            "vram_percent": memory_info.get("vram_percent", 0),
            "models_loaded": sum(1 for m in self.models.values() if m.loaded),
        }

    def unload_all(self) -> None:
        """Descarrega todos os modelos."""
        for model in self.models.values():
            if model.loaded:
                model.unload()

        # Limpar registro de último acesso
        self.last_used.clear()

        # Forçar limpeza de memória
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc

        gc.collect()

        self.logger.info("Todos os modelos foram descarregados")

    def __del__(self) -> None:
        """Descarregar modelos quando o gerenciador for destruído."""
        try:
            self.unload_all()
        except:
            # Ignorar erros no destrutor
            pass
