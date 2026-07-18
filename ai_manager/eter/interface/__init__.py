#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pacote de interfaces para o assistente Èter

Este pacote contém diferentes interfaces para interagir com o assistente
Èter, incluindo linha de comando (CLI), API e potencialmente interfaces
gráficas no futuro.
"""

from cerberus_api.pipeline_builder.ai_manager.eter.interface.cli import (
    EterCLI,
    initialize,
)

__all__ = ["EterCLI", "initialize"]
