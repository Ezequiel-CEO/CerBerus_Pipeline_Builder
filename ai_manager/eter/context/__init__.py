#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de gestão de contexto para o assistente Èter.

Fornece componentes para armazenamento e recuperação de:
- Memória de curto e longo prazo
- Conhecimento técnico
- Contexto de execução
"""

from cerberus_api.pipeline_builder.ai_manager.eter.context.memory import EterMemory

__all__ = [
    "EterMemory",
]
