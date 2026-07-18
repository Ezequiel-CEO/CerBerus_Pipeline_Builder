#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilitários para o PipelineBuilder.

Este pacote contém funções auxiliares para processamento de
conteúdo, armazenamento de dados e operações gerais.
"""

from cerberus_api.pipeline_builder.utils.storage import (
    save_content_chunk,
    save_analysis_result,
    save_pipeline,
    load_pipeline,
    load_content_chunk,
    load_analysis_result,
)

from cerberus_api.pipeline_builder.utils.nlp_utils import (
    preprocess_text,
    extract_keywords,
    extract_named_entities,
    calculate_text_similarity,
    generate_text_summary,
    detect_language,
    hash_text,
    split_into_chunks,
)

__all__ = [
    "save_content_chunk",
    "save_analysis_result",
    "save_pipeline",
    "load_pipeline",
    "load_content_chunk",
    "load_analysis_result",
    "preprocess_text",
    "extract_keywords",
    "extract_named_entities",
    "calculate_text_similarity",
    "generate_text_summary",
    "detect_language",
    "hash_text",
    "split_into_chunks",
]
