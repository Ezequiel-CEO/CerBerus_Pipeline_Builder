#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Èter - Módulo de Agente Inteligente do CerBerus FMK.

Este módulo implementa o agente Èter, um assistente inteligente integrado
ao framework, responsável por automatizar tarefas, auxiliar o usuário e executar
operações complexas.

O Èter é o núcleo de inteligência que pode operar o CerBerus FMK mesmo sem
intervenção humana, funcionando como um "piloto automático" do sistema.

Versão: 0.1.0
"""

import logging
import os
from typing import Optional, Dict, Any, List

# Configurar logging do módulo
logger = logging.getLogger("eter")

# Instância global do agente
_eter_instance = None


def initialize(
    user_id: str = "default",
    model_dir: Optional[str] = None,
    knowledge_dir: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    framework_integration: bool = True,
    **kwargs,
) -> Any:
    """
    Inicializa o agente Èter.

    Args:
        user_id: Identificador do usuário
        model_dir: Diretório de modelos
        knowledge_dir: Diretório de base de conhecimento
        workspace_dir: Diretório de trabalho
        framework_integration: Ativar integração com o framework CerBerusFMK
        **kwargs: Parâmetros adicionais

    Returns:
        Instância do agente Èter
    """
    global _eter_instance

    logger.info(f"Inicializando Èter para usuário: {user_id}")

    try:
        from cerberus_api.pipeline_builder.ai_manager.eter.agent import EterAgent

        # Se já existe uma instância para este usuário, retorná-la
        if _eter_instance is not None and _eter_instance.user_id == user_id:
            logger.info(f"Reutilizando instância existente para usuário: {user_id}")
            return _eter_instance

        # Configurar diretórios padrão se não especificados
        if model_dir is None:
            model_dir = os.environ.get("ETER_MODEL_DIR", os.path.expanduser("~/models"))

        if knowledge_dir is None:
            knowledge_dir = os.environ.get(
                "ETER_KNOWLEDGE_DIR", os.path.expanduser("~/knowledge")
            )

        if workspace_dir is None:
            workspace_dir = os.environ.get("ETER_WORKSPACE_DIR", os.getcwd())

        # Criar a instância do agente
        _eter_instance = EterAgent(
            user_id=user_id,
            model_dir=model_dir,
            knowledge_dir=knowledge_dir,
            workspace_dir=workspace_dir,
            enable_framework_bridge=framework_integration,
            **kwargs,
        )

        logger.info(
            f"Èter inicializado. Integração com framework: {framework_integration}"
        )
        return _eter_instance

    except Exception as e:
        logger.error(f"Erro ao inicializar Èter: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def create_eter_instance(
    enable_framework_bridge: bool = True,
    register_components: bool = True,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    user_id: Optional[str] = None,
    verbose: bool = False,
) -> Any:
    """
    Cria uma instância do Èter com configurações simplificadas.

    Esta função facilita a criação de uma instância do Èter com opções comuns
    e registra componentes do framework quando solicitado.

    Args:
        enable_framework_bridge: Se deve ativar a ponte com o framework
        register_components: Se deve registrar componentes do framework
        model_name: Nome do modelo a ser carregado
        model_path: Caminho para o arquivo do modelo (opcional)
        user_id: ID do usuário para personalização
        verbose: Se deve habilitar logs detalhados

    Returns:
        Instância configurada do EterAgent
    """
    log_level = "DEBUG" if verbose else "INFO"
    logging.getLogger("eter").setLevel(getattr(logging, log_level))

    # Inicializar o agente
    agent = initialize(
        user_id=user_id or "default",
        framework_integration=enable_framework_bridge,
        verbose=verbose,
    )

    if not agent:
        logger.error("Falha ao inicializar o agente Èter")
        raise RuntimeError("Falha ao inicializar o agente Èter")

    # Registrar componentes do framework
    if (
        enable_framework_bridge
        and register_components
        and hasattr(agent, "framework_bridge")
    ):
        try:
            capabilities = agent.framework_bridge.discover_framework_capabilities()
            components_count = sum(len(items) for items in capabilities.values())
            logger.info(f"Componentes registrados: {components_count}")
        except Exception as e:
            logger.error(f"Erro ao registrar componentes: {e}")

    # Carregar modelo se especificado
    if model_name and hasattr(agent, "set_active_model"):
        try:
            agent.set_active_model(model_name, force_reload=False)
            logger.info(f"Modelo ativo definido: {model_name}")
        except Exception as e:
            logger.error(f"Erro ao definir modelo ativo: {e}")

    return agent


from cerberus_api.pipeline_builder.ai_manager.eter.agent import EterAgent

__all__ = ["EterAgent", "initialize", "create_eter_instance"]

"""
Módulo ETER - Loaders otimizados para modelos GGUF e outros formatos.
Este módulo fornece implementações de alta performance para carregar e utilizar
modelos de linguagem em diferentes formatos.
"""

from cerberus_api.pipeline_builder.ai_manager.eter.agent import EterAgent
from .base_loader import BaseModelLoader
from .llama_cpp_loader import LlamaCppLoader

# Funções de inicialização do agente
try:
    from cerberus_api.pipeline_builder.ai_manager.eter.agent import (
        initialize,
        create_eter_instance,
    )
except ImportError:
    # Se não for possível importar, definir funções vazias
    def initialize(*args, **kwargs):
        pass

    def create_eter_instance(*args, **kwargs):
        pass


__all__ = [
    "EterAgent",
    "initialize",
    "create_eter_instance",
    "BaseModelLoader",
    "LlamaCppLoader",
]

"""
Éter - Sistema de Aceleração e Inferência para o CerBerus FMK

Este pacote contém implementações para aceleração de operações críticas no CerBerus FMK,
incluindo o EterCUDA Accelerator para operações em GPU usando CUDA e loaders de modelos.

Author: CerBerus FMK Team
"""

# Exportações de loaders de modelos
__all__ = ["BaseModelLoader", "LlamaCppLoader"]

# Tenta importar o acelerador CUDA
try:
    from .cuda_accelerator import EterCUDAAccelerator, integrate_cuda_accelerator

    __all__.extend(["EterCUDAAccelerator", "integrate_cuda_accelerator"])
except ImportError:
    # Define versões mock caso as importações falhem
    def integrate_cuda_accelerator(engine):
        print("AVISO: EterCUDA Accelerator não disponível")
        return None

    class EterCUDAAccelerator:
        def __init__(self, *args, **kwargs):
            print("AVISO: EterCUDA Accelerator não disponível")
            self.is_initialized = False

    __all__.extend(["EterCUDAAccelerator", "integrate_cuda_accelerator"])
