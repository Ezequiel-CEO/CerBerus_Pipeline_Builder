#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ModelHub - Gerenciador Central de Modelos
======================================

Gerencia o carregamento, cache e uso de modelos no CerBerusFMK.
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import threading
from pathlib import Path

from cerberus_api.utils.logging_config import get_logger

# Configuração de logging
logger = get_logger("model_hub")


class ModelType(Enum):
    """Tipos de modelos suportados pelo sistema."""

    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    GENERATION = "generation"
    CUSTOM = "custom"


@dataclass
class ModelVersion:
    """Informações sobre uma versão específica de modelo."""

    version: str
    created_at: str
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    file_path: Optional[str] = None
    config_path: Optional[str] = None
    is_active: bool = False


@dataclass
class ModelInfo:
    """Informações detalhadas sobre um modelo."""

    name: str
    model_type: ModelType
    description: str = ""
    source: str = "local"
    framework: str = "pytorch"
    requires_gpu: bool = False
    input_type: str = "text"
    output_type: str = "text"
    versions: List[ModelVersion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    active_version: Optional[str] = None


class ModelHub:
    """
    Hub central para gerenciamento de modelos de IA.

    Responsabilidades:
    - Carregar e gerenciar modelos
    - Manter registro de versões
    - Otimizar uso de memória e GPU
    - Fornecer modelos sob demanda
    """

    def __init__(self, models_dir: Optional[str] = None):
        """Inicializa o ModelHub."""
        self.models_dir = models_dir or os.path.join(
            os.path.dirname(__file__), "../../../../models"
        )
        self.models_dir = os.path.abspath(self.models_dir)

        # Criar diretório se não existir
        os.makedirs(self.models_dir, exist_ok=True)

        # Cache de modelos
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Dict] = {}

        # Carregar catálogo
        self._load_catalog()

        logger.info(f"ModelHub inicializado. Diretório: {self.models_dir}")
        logger.info(f"Modelos disponíveis: {len(self.model_configs)}")

    def _load_catalog(self):
        """Carrega o catálogo de modelos."""
        catalog_file = os.path.join(self.models_dir, "catalog.json")

        if os.path.exists(catalog_file):
            try:
                with open(catalog_file, "r") as f:
                    self.model_configs = json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar catálogo: {e}")
                self.model_configs = {}

        # Escanear diretório por modelos
        self._scan_models_directory()

        logger.info(f"Catálogo de modelos carregado: {len(self.model_configs)} modelos")

    def _scan_models_directory(self):
        """Escaneia o diretório de modelos."""
        for root, _, files in os.walk(self.models_dir):
            for file in files:
                if file.endswith(
                    (".gguf", ".ggml", ".bin", ".pt", ".pth", ".safetensors")
                ):
                    model_path = os.path.join(root, file)
                    model_id = os.path.splitext(file)[0]

                    if model_id not in self.model_configs:
                        self.model_configs[model_id] = {
                            "path": model_path,
                            "format": os.path.splitext(file)[1][1:],
                            "size": os.path.getsize(model_path),
                            "loaded": False,
                        }

    def get_available_models(self) -> List[str]:
        """Retorna lista de modelos disponíveis."""
        return list(self.model_configs.keys())

    def load_model(self, model_id: str) -> Optional[Any]:
        """Carrega um modelo específico."""
        if model_id not in self.model_configs:
            logger.error(f"Modelo não encontrado: {model_id}")
            return None

        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        try:
            config = self.model_configs[model_id]
            model_path = config["path"]
            model_format = config["format"]

            # Carregar baseado no formato
            if model_format == "gguf":
                from ...inference_engine import native_gguf_runner

                model = native_gguf_runner.load_model(model_path)
            elif model_format in ["pt", "pth", "bin"]:
                import torch

                model = torch.load(model_path)
            else:
                logger.error(f"Formato não suportado: {model_format}")
                return None

            self.loaded_models[model_id] = model
            config["loaded"] = True
            return model

        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_id}: {e}")
            return None

    def unload_model(self, model_id: str) -> bool:
        """Descarrega um modelo da memória."""
        if model_id in self.loaded_models:
            try:
                del self.loaded_models[model_id]
                self.model_configs[model_id]["loaded"] = False
                return True
            except Exception as e:
                logger.error(f"Erro ao descarregar modelo {model_id}: {e}")
        return False

    def unload_all_models(self):
        """Descarrega todos os modelos."""
        for model_id in list(self.loaded_models.keys()):
            self.unload_model(model_id)

    def __del__(self):
        """Cleanup ao destruir o objeto."""
        try:
            self.unload_all_models()
        except:
            pass

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Obtém informações sobre um modelo.

        Args:
            model_name: Nome do modelo

        Returns:
            Informações do modelo ou None se não encontrado
        """
        return self.model_configs.get(model_name)

    def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelInfo]:
        """
        Lista modelos disponíveis, opcionalmente filtrados por tipo.

        Args:
            model_type: Filtrar por tipo de modelo

        Returns:
            Lista de informações de modelos
        """
        if model_type is None:
            return list(self.model_configs.values())

        return [
            model
            for model in self.model_configs.values()
            if model.get("model_type") == model_type
        ]

    def register_model(self, model_info: ModelInfo) -> bool:
        """
        Registra um novo modelo no hub.

        Args:
            model_info: Informações do modelo

        Returns:
            True se o registro foi bem-sucedido
        """
        # Verificar se já existe
        if model_info.name in self.model_configs:
            logger.warning(f"Modelo já existe: {model_info.name}")
            return False

        # Adicionar ao dicionário
        self.model_configs[model_info.name] = model_info

        # Salvar catálogo
        self._save_catalog()

        logger.info(f"Modelo registrado: {model_info.name}")
        return True

    def add_model_version(self, model_name: str, version: ModelVersion) -> bool:
        """
        Adiciona uma nova versão a um modelo existente.

        Args:
            model_name: Nome do modelo
            version: Informações da versão

        Returns:
            True se a adição foi bem-sucedida
        """
        if model_name not in self.model_configs:
            logger.warning(f"Modelo não encontrado: {model_name}")
            return False

        model_info = self.model_configs[model_name]

        # Verificar se a versão já existe
        existing_versions = [v.version for v in model_info.versions]
        if version.version in existing_versions:
            logger.warning(f"Versão já existe: {model_name} v{version.version}")
            return False

        # Adicionar versão
        model_info.versions.append(version)

        # Se for a primeira versão ou marcada como ativa, definir como ativa
        if not model_info.active_version or version.is_active:
            model_info.active_version = version.version

            # Desativar outras versões se esta for ativa
            if version.is_active:
                for v in model_info.versions:
                    if v.version != version.version:
                        v.is_active = False

        # Salvar catálogo
        self._save_catalog()

        logger.info(f"Versão adicionada: {model_name} v{version.version}")
        return True

    def set_active_version(self, model_name: str, version: str) -> bool:
        """
        Define a versão ativa de um modelo.

        Args:
            model_name: Nome do modelo
            version: Versão a ser ativada

        Returns:
            True se a operação foi bem-sucedida
        """
        if model_name not in self.model_configs:
            logger.warning(f"Modelo não encontrado: {model_name}")
            return False

        model_info = self.model_configs[model_name]

        # Verificar se a versão existe
        version_exists = any(v.version == version for v in model_info.versions)
        if not version_exists:
            logger.warning(f"Versão não encontrada: {model_name} v{version}")
            return False

        # Definir como ativa
        model_info.active_version = version

        # Atualizar flags de ativo nas versões
        for v in model_info.versions:
            v.is_active = v.version == version

        # Salvar catálogo
        self._save_catalog()

        # Descarregar versão anteriormente carregada
        if model_name in self.loaded_models:
            logger.info(f"Descarregando modelo anterior: {model_name}")
            self.loaded_models.pop(model_name, None)

        logger.info(f"Versão ativa definida: {model_name} v{version}")
        return True

    def get_model(self, model_name: str) -> Any:
        """
        Obtém uma instância carregada do modelo.

        Args:
            model_name: Nome do modelo

        Returns:
            Instância do modelo ou None se não encontrado/carregado
        """
        # Verificar se o modelo já está carregado
        if model_name in self.loaded_models:
            logger.debug(f"Usando modelo já carregado: {model_name}")
            return self.loaded_models[model_name]

        # Verificar se o modelo existe
        if model_name not in self.model_configs:
            logger.warning(f"Modelo não encontrado: {model_name}")
            return None

        # Obter informações do modelo
        model_info = self.model_configs[model_name]

        # Verificar se há uma versão ativa
        if not model_info.active_version:
            logger.warning(f"Modelo não tem versão ativa: {model_name}")
            return None

        # Obter informações da versão ativa
        active_version = next(
            (v for v in model_info.versions if v.version == model_info.active_version),
            None,
        )

        if not active_version:
            logger.warning(
                f"Versão ativa não encontrada: {model_name} v{model_info.active_version}"
            )
            return None

        # Verificar se há caminho para o arquivo do modelo
        if not active_version.file_path:
            logger.warning(
                f"Caminho do modelo não definido: {model_name} v{active_version.version}"
            )
            return None

        # Carregar modelo
        try:
            # TODO: Implementar carregamento real do modelo baseado no framework
            # Por enquanto, apenas simulamos
            model = f"MODELO_{model_name}_{active_version.version}"

            # Armazenar no cache
            self.loaded_models[model_name] = model

            logger.info(f"Modelo carregado: {model_name} v{active_version.version}")
            return model

        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_name}: {e}")
            return None

    def is_model_registered(self, model_name: str) -> bool:
        """
        Verifica se um modelo está registrado no catálogo.

        Args:
            model_name: Nome do modelo para verificar

        Returns:
            True se o modelo estiver registrado
        """
        # Verificar catálogo
        if not os.path.exists(os.path.join(self.models_dir, "catalog.json")):
            return False

        try:
            with open(os.path.join(self.models_dir, "catalog.json"), "r") as f:
                catalog = json.load(f)

            if not isinstance(catalog, dict) or "models" not in catalog:
                return False

            # Verificar modelos existentes
            models_data = catalog["models"]
            if not isinstance(models_data, dict):
                return False

            for model_id, data in models_data.items():
                # Verificar pelo nome ou ID
                if model_name == model_id or model_name == data.get("name", ""):
                    # Verificar se o arquivo existe
                    file_path = data.get("path", "")
                    if os.path.exists(file_path):
                        return True

            return False

        except Exception as e:
            logger.error(f"Erro ao verificar modelo '{model_name}': {str(e)}")
            return False

    def add_model(
        self,
        model_path: str,
        model_type: str = "gguf",
        model_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Adiciona um modelo ao catálogo e o torna disponível para uso.

        Args:
            model_path: Caminho para o arquivo do modelo
            model_type: Tipo do modelo (gguf, onnx, etc)
            model_info: Informações adicionais sobre o modelo

        Returns:
            bool: True se o modelo foi adicionado com sucesso
        """
        # Extrair ID do modelo do nome do arquivo
        model_id = os.path.basename(model_path)
        # Remover extensão
        if model_id.lower().endswith(".gguf"):
            model_id = model_id[:-5]
        if model_id.lower().endswith(".final"):
            model_id = model_id[:-6]

        # Normalizar ID
        model_id = model_id.lower().replace(" ", "_")

        self.logger.info(
            f"Adicionando modelo: {model_id} ({model_type}) em {model_path}"
        )

        # Verificar se o arquivo existe
        if not os.path.exists(model_path):
            self.logger.error(f"Arquivo de modelo não encontrado: {model_path}")
            return False

        # Verificar se já existe um modelo com esse ID
        for model in self.get_available_models():
            if model.get("id") == model_id:
                self.logger.warning(
                    f"Já existe um modelo com ID {model_id}. Atualizando."
                )
                self.model_configs.pop(model_id)
                break

        # Preparar a informação do modelo
        model_data = {
            "id": model_id,
            "name": model_id,  # Usar o mesmo ID como nome
            "path": model_path,
            "type": model_type,
            "loaded": False,
        }

        # Adicionar informações extras
        if model_info:
            model_data.update(model_info)

        # Adicionar ao catálogo
        self.model_configs[model_id] = model_data

        # Salvar o catálogo atualizado
        self._save_catalog()

        self.logger.info(f"Modelo {model_id} adicionado com sucesso")
        return True

    def _save_catalog(self) -> None:
        """Salva o catálogo de modelos no disco."""
        try:
            # Converter objetos para dicionários
            models_data = []

            for model_id, data in self.model_configs.items():
                model_data = {
                    "id": model_id,
                    "name": data.get("name", model_id),
                    "path": data.get("path", ""),
                    "type": data.get("type", ""),
                    "loaded": data.get("loaded", False),
                    "size": data.get("size", 0),
                    "size_gb": data.get("size", 0) / (1024**3),
                }

                models_data.append(model_data)

            catalog_data = {"version": "1.0", "models": models_data}

            with open(
                os.path.join(self.models_dir, "catalog.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(catalog_data, f, ensure_ascii=False, indent=2)

            logger.info("Catálogo de modelos salvo com sucesso")

        except Exception as e:
            logger.error(f"Erro ao salvar catálogo de modelos: {e}")

    def generate_text(
        self,
        prompt: str,
        model_name: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        streaming: bool = False,
    ):
        """
        Gera texto a partir de um prompt usando um modelo específico.

        Args:
            prompt: Texto de entrada para o modelo
            model_name: Nome do modelo a ser usado (usa o último carregado se None)
            max_tokens: Número máximo de tokens a gerar
            temperature: Controla aleatoriedade (0.0=determinístico, 1.0=criativo)
            streaming: Se True, retorna um gerador para streaming de tokens

        Returns:
            Texto gerado ou um gerador de tokens se streaming=True
        """
        try:
            # Verificar se temos algum modelo carregado
            if not model_name:
                # Se não foi especificado, usar o último modelo carregado
                if not self.loaded_models or len(self.loaded_models) == 0:
                    logger.error("Nenhum modelo carregado para geração de texto")
                    return "Erro: Nenhum modelo está carregado. Use add_model() e load_model() primeiro."

                # Usar o primeiro modelo carregado
                model_name = list(self.loaded_models.keys())[0]

            # Verificar se o modelo específico está carregado
            if model_name not in self.loaded_models:
                # Tentar carregar o modelo se ele existe no catálogo
                for model in self.get_available_models():
                    if model.get("name") == model_name:
                        success = self.load_model(model_name)
                        if not success:
                            logger.error(f"Falha ao carregar modelo '{model_name}'")
                            return f"Erro: Não foi possível carregar o modelo '{model_name}'."
                        break
                else:
                    # Modelo não encontrado
                    logger.error(f"Modelo '{model_name}' não encontrado no catálogo")
                    return f"Erro: Modelo '{model_name}' não encontrado."

            try:
                # Obter a implementação apropriada
                gguf_impl = self._get_best_gguf_implementation()

                if not gguf_impl:
                    logger.error("Nenhuma implementação GGUF disponível")
                    return "Erro: Nenhuma implementação de LLM disponível."

                # Obter o objeto do modelo
                if hasattr(self.loaded_models[model_name], "model"):
                    model = self.loaded_models[model_name].model
                else:
                    model = self.loaded_models[model_name]

                # Se estamos usando a implementação simulada
                if gguf_impl.__name__ == "SimulatedGGUF":
                    if streaming:
                        # Implementação básica para streaming simulado
                        def token_generator():
                            response = f"[SIMULAÇÃO] Resposta para: {prompt}"
                            for char in response:
                                yield char
                                import time

                                time.sleep(0.01)  # Simular delay entre tokens

                        return token_generator()
                    else:
                        return f"[SIMULAÇÃO] Resposta para: {prompt}"

                # Verificar se a implementação tem o método de geração
                if hasattr(gguf_impl, "generate"):
                    # Gerar texto
                    if streaming:
                        return gguf_impl.generate(
                            model=model,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            streaming=True,
                        )
                    else:
                        return gguf_impl.generate(
                            model=model,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            streaming=False,
                        )
                elif hasattr(model, "generate"):
                    # Usar o método do próprio modelo
                    if streaming:
                        return model.generate(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            streaming=True,
                        )
                    else:
                        return model.generate(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            streaming=False,
                        )
                else:
                    # Fallback para quando o método de geração não é encontrado
                    logger.error("Método de geração não encontrado")
                    return f"Erro: Interface de geração não disponível para o modelo '{model_name}'."

            except Exception as e:
                logger.error(f"Erro ao gerar texto com o modelo '{model_name}': {e}")
                import traceback

                logger.error(traceback.format_exc())
                return f"Erro na geração de texto: {str(e)}"

        except Exception as e:
            logger.error(f"Erro ao preparar modelo para geração: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return f"Erro: {str(e)}"

    def is_model_loaded(self, model_name: str) -> bool:
        """
        Verifica se um modelo específico está carregado.

        Args:
            model_name: Nome do modelo para verificar

        Returns:
            bool: True se o modelo está carregado, False caso contrário
        """
        if not model_name:
            return False

        # Verificar se o modelo está no dicionário de modelos carregados
        return (
            model_name in self.loaded_models
            and self.loaded_models[model_name] is not None
        )

    def _get_best_gguf_implementation(self):
        """
        Obtém a melhor implementação GGUF disponível.

        Returns:
            Implementação GGUF ou None se nenhuma estiver disponível
        """
        # Verificar implementações disponíveis
        implementations = []

        # GGUF
        try:
            from cerberus_api.pipeline_builder.ai_manager.eter.gguf_loader import (
                GGUFModel,
                GGUFConfig,
            )

            implementations.append(("GGUF", GGUFModel))
        except ImportError as e:
            logger.warning(f"GGUF não disponível: {e}")

        # Se nenhuma implementação estiver disponível, retornar None
        if not implementations:
            return None

        # Retornar a primeira implementação disponível
        return implementations[0][1]
