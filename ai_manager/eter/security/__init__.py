#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de segurança do Èter

Este módulo implementa funcionalidades de segurança para o assistente Èter,
incluindo controle de acesso, validação de permissões e operações seguras.
"""

from cerberus_api.pipeline_builder.ai_manager.eter.security.access_control import (
    AccessControl,
)

__all__ = ["AccessControl"]
