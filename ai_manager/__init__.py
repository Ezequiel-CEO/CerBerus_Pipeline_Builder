#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo AI Manager - Gerenciamento de modelos de IA para o framework CerBerusFMK.

Módulo de extensão para gerenciar o ciclo completo de modelos de IA:
- Carregamento e gerenciamento de modelos
- Fine-tuning a partir de pipelines de dados
- Inferência otimizada e eficiente
- Aprendizado contínuo e adaptação
"""

__version__ = "0.1.0"

# Importações principais
from cerberus_api.pipeline_builder.ai_manager.model_hub import (
    ModelHub,
    ModelInfo,
    ModelType,
    ModelVersion,
)

# Versão simplificada para uso inicial
model_hub = ModelHub()

# Exportar símbolos para namespace do pacote
__all__ = [
    "ModelHub",
    "ModelInfo",
    "ModelType",
    "ModelVersion",
    "model_hub",
]

# Disponibilizar submódulos
from . import eter

__all__ += ["eter"]

from .unified_engine import UnifiedEngine

__all__ += ["UnifiedEngine"]
