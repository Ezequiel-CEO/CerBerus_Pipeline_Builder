#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo para carregar e gerenciar modelos LLM em formato GGUF.

Este módulo fornece uma interface para carregar e usar modelos de
linguagem baseados em GGUF, como o Qwen 7B, otimizando uso de recursos.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Generator

import torch
import numpy as np
from tqdm import tqdm

from cerberus_api.utils.logging_config import get_logger
from cerberus_api.pipeline_builder.utils.memory_manager import MemoryManager
from cerberus_api.pipeline_builder.ai_manager.eter.llm.prompt_formatter import (
    get_formatter_for_model,
)

# Definir variável global para controle de disponibilidade
NATIVE_GGUF_AVAILABLE = False

# Tentar importar nossa implementação nativa GGUF do CerBerusFMK
try:
    from libs.CerBerusFMK.CerBerusFMK.llm_local.native_gguf import NativeGGUFRunner
    from libs.CerBerusFMK.CerBerusFMK.inference_engine.engine import InferenceEngine

    NATIVE_GGUF_AVAILABLE = True
except ImportError as e:
    print(f"Implementação nativa GGUF não disponível: {e}")
    print("Modelos GGUF terão funcionalidade limitada")

# Configuração de logging
logger = get_logger("eter_llm")


class GGUFModel:
    """
    Classe para gerenciar modelos no formato GGUF (sucessor do GGML).

    Esta classe facilita o carregamento e uso de modelos como o Qwen 7B
    em dispositivos com diferentes capacidades de hardware.
    """

    def __init__(
        self,
        model_path: str,
        model_config: Optional[Dict[str, Any]] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 significa todas as camadas possíveis na GPU
        use_mmaps: bool = True,
        use_memory_manager: bool = True,
        verbose: bool = False,
    ):
        """
        Inicializa o modelo GGUF.

        Args:
            model_path: Caminho para o arquivo de modelo GGUF
            model_config: Configuração adicional para o modelo
            n_ctx: Tamanho do contexto (janela de tokens)
            n_gpu_layers: Número de camadas para offload para GPU (-1 = todas possíveis)
            use_mmaps: Usar memory-mapped files para economizar RAM
            use_memory_manager: Usar gerenciador de memória para otimização
            verbose: Modo detalhado para debug
        """
        self.logger = logger
        self.model_path = os.path.abspath(model_path)
        self.model_config = model_config or {}
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.use_mmaps = use_mmaps
        self.verbose = verbose

        # Verificar se o modelo existe
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Arquivo de modelo não encontrado: {self.model_path}"
            )

        # Configurar gerenciador de memória
        self.memory_manager = MemoryManager() if use_memory_manager else None

        # O modelo será carregado sob demanda
        self.model = None
        self.loaded = False
        self.last_used = 0

        # Informações sobre o modelo
        self.model_info = self._get_model_info()

        self.logger.info(f"GGUFModel inicializado: {os.path.basename(model_path)}")

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
            "n_ctx": self.n_ctx,
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
            self.last_used = time.time()
            return True

        if not NATIVE_GGUF_AVAILABLE:
            self.logger.error("Implementação nativa GGUF não está disponível")
            return False

        try:
            self.logger.info(f"Carregando modelo: {self.model_path}")

            # Obter número de camadas GPU com base na memória disponível
            n_gpu_layers = self.n_gpu_layers
            if self.memory_manager is not None and n_gpu_layers == -1:
                available_vram = self._get_available_vram()
                model_size = self.model_info.get("size_gb", 0)

                if available_vram > 0:
                    # Regra heurística: garantir que pelo menos 2/3 do modelo caiba na VRAM
                    if available_vram > model_size * 0.67:
                        n_gpu_layers = -1  # Todas as camadas na GPU
                    else:
                        # Estimativa aproximada baseada no tamanho total
                        n_gpu_layers = int((available_vram / model_size) * 32)
                else:
                    n_gpu_layers = 0  # CPU apenas

            self.logger.info(f"Usando {n_gpu_layers} camadas GPU")

            # Carregar o modelo
            self.model = NativeGGUFRunner(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=self.verbose,
            )

            self.loaded = True
            self.last_used = time.time()
            self.logger.info(f"Modelo {self.model_info['name']} carregado com sucesso")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            self.model = None
            self.loaded = False
            return False

    def _get_available_vram(self) -> float:
        """
        Obtém a quantidade disponível de VRAM em GB.

        Returns:
            Quantidade disponível de VRAM em GB
        """
        try:
            if self.memory_manager is None:
                return 0.0

            memory_info = self.memory_manager.get_memory_usage()
            vram_total = memory_info.get("vram_total_mb", 0) / 1024  # Converter para GB
            vram_used = memory_info.get("vram_used_mb", 0) / 1024  # Converter para GB

            available_vram = max(0, vram_total - vram_used)
            return available_vram
        except Exception as e:
            self.logger.warning(f"Erro ao obter VRAM disponível: {e}")
            return 0.0

    def unload(self) -> bool:
        """
        Descarrega o modelo da memória.

        Returns:
            True se descarregado com sucesso
        """
        if not self.loaded or self.model is None:
            return True

        try:
            # Liberar explicitamente o modelo
            del self.model
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

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        streaming: bool = False,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Gera texto a partir do prompt fornecido.

        Args:
            prompt: Texto de entrada
            max_tokens: Número máximo de tokens a gerar
            temperature: Temperatura para amostragem (0.0-1.0)
            top_p: Valor para amostragem núcleo (0.0-1.0)
            top_k: Número de tokens mais prováveis a considerar
            repeat_penalty: Penalidade para repetição
            streaming: Se True, retorna um gerador para streaming
            system_prompt: Instrução do sistema a incluir (opcional)
            chat_history: Histórico de conversas (opcional)

        Returns:
            str ou Generator: Texto gerado ou gerador para streaming
        """
        try:
            # Garantir que o modelo está carregado
            if not self.loaded:
                self.load()

            # Atualizar timestamp de último uso
            self.last_used = time.time()

            # Verificar se devemos usar formatador específico
            model_basename = os.path.basename(self.model_path).lower()
            formatter = get_formatter_for_model(model_basename)

            # Formatar o prompt se temos system_prompt ou chat_history
            if system_prompt or chat_history:
                prompt = formatter.format_prompt(
                    user_input=prompt,
                    system_prompt=system_prompt,
                    chat_history=chat_history,
                )

            # Configurar parâmetros de geração
            params = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "stream": streaming,
            }

            # Gerar a resposta
            if streaming:
                # Retorna um gerador
                response_generator = self.model(prompt=prompt, **params)
                return (chunk["choices"][0]["text"] for chunk in response_generator)
            else:
                # Retorna a resposta completa
                response = self.model(prompt=prompt, **params)
                return response["choices"][0]["text"]

        except Exception as e:
            self.logger.error(f"Erro na geração: {e}")
            if streaming:
                return (f"Erro: {e}" for _ in range(1))
            else:
                return f"Erro: {e}"

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Obtém informações sobre uso de memória.

        Returns:
            Dicionário com informações de uso de memória
        """
        usage = {}

        if self.memory_manager is not None:
            ram_usage, vram_usage = self.memory_manager.get_memory_usage()
            usage["ram_mb"] = ram_usage
            usage["vram_mb"] = vram_usage

        return usage

    def __str__(self) -> str:
        """Representação em string do modelo."""
        status = "Carregado" if self.loaded else "Não carregado"
        return f"GGUF Model: {self.model_info['name']} ({status})"


class GGUFManager:
    """
    Gerenciador para múltiplos modelos GGUF.

    Permite carregar vários modelos e gerenciar uso de memória
    carregando/descarregando conforme necessário.
    """

    def __init__(
        self,
        models_dir: str = "/home/agressor/.cerberusfmk/models",
        max_models_loaded: int = 1,
        memory_threshold: float = 0.8,  # Threshold de memória (0.0-1.0)
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
        self.models_dir = os.path.abspath(models_dir)
        self.max_models_loaded = max_models_loaded
        self.memory_threshold = memory_threshold
        self.auto_unload = auto_unload

        # Gerenciador de memória
        self.memory_manager = MemoryManager()

        # Dicionário de modelos
        self.models: Dict[str, GGUFModel] = {}

        # Criar diretório se não existir
        os.makedirs(self.models_dir, exist_ok=True)

        # Descobrir modelos disponíveis
        self._discover_models()

        self.logger.info(
            f"GGUFManager inicializado. Modelos disponíveis: {len(self.models)}"
        )

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
                model = GGUFModel(
                    model_path=file_path, use_memory_manager=True, verbose=False
                )
                self.models[model_name] = model
                self.logger.info(f"Modelo registrado: {model_name}")
            except Exception as e:
                self.logger.error(f"Erro ao registrar modelo {model_name}: {e}")

    def add_model(
        self,
        model_path: str,
        model_config: Optional[Dict[str, Any]] = None,
        load_immediately: bool = False,
    ) -> Optional[str]:
        """
        Adiciona um novo modelo ao gerenciador.

        Args:
            model_path: Caminho para o arquivo de modelo GGUF
            model_config: Configuração adicional para o modelo
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
            model = GGUFModel(
                model_path=model_path,
                model_config=model_config,
                use_memory_manager=True,
                verbose=False,
            )

            # Adicionar ao dicionário
            self.models[model_name] = model

            # Carregar o modelo se solicitado
            if load_immediately:
                model.load()

            self.logger.info(f"Modelo adicionado: {model_name}")
            return model_name

        except Exception as e:
            self.logger.error(f"Erro ao adicionar modelo: {e}")
            return None

    def get_model(self, model_name: str) -> Optional[GGUFModel]:
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
        return self.models[model_name].load()

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

        self.logger.info(f"Modelo removido: {model_name}")
        return True

    def _manage_memory(self) -> None:
        """
        Gerencia a memória, descarregando modelos quando necessário.
        """
        if not self.auto_unload:
            return

        # Contar modelos carregados
        loaded_models = [m for m in self.models.values() if m.loaded]

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
            loaded_models.sort(key=lambda m: m.last_used)

            # Descarregar modelos até ficar abaixo dos limites
            for model in loaded_models[: -self.max_models_loaded]:
                model_name = model.model_info["name"]
                self.logger.info(
                    f"Descarregando modelo {model_name} para economizar memória"
                )
                model.unload()

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
                "n_ctx": model.n_ctx,
            }
            models_info.append(info)

        return models_info

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

        # Forçar limpeza de memória
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc

        gc.collect()

        self.logger.info("Todos os modelos foram descarregados")

    def __del__(self) -> None:
        """Descarregar modelos quando o gerenciador for destruído."""
        self.unload_all()
