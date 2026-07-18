#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analisadores de conteúdo para o PipelineBuilder.

Este pacote contém os analisadores para avaliar relevância,
extrair tópicos e categorizar conteúdo.
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Garantir que as dependências do NLTK estejam disponíveis
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

from cerberus_api.pipeline_builder.analyzers.content_analyzer import ContentAnalyzer
from cerberus_api.pipeline_builder.analyzers.enhanced_analyzer import EnhancedAnalyzer

__all__ = ["ContentAnalyzer", "EnhancedAnalyzer", "word_tokenize", "sent_tokenize"]
