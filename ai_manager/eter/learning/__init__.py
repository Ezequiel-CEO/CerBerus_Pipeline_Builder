#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de aprendizado do Èter

Este módulo implementa as funcionalidades para aprendizado incremental
do assistente inteligente, incluindo observação de ações, feedback do
usuário e adaptação baseada em contexto.
"""

from cerberus_api.pipeline_builder.ai_manager.eter.learning.observer import EterObserver
from cerberus_api.pipeline_builder.ai_manager.eter.learning.feedback import EterFeedback

__all__ = ["EterObserver", "EterFeedback"]
