#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coletores de conteúdo para o PipelineBuilder.

Este pacote contém os coletores para diferentes tipos de fontes,
como web, GitHub, PDFs e YouTube.
"""

from cerberus_api.pipeline_builder.collectors.web_collector import WebCollector
from cerberus_api.pipeline_builder.collectors.pdf_collector import PDFCollector
from cerberus_api.pipeline_builder.collectors.github_collector import GitHubCollector
from cerberus_api.pipeline_builder.collectors.youtube_collector import YouTubeCollector

__all__ = ["WebCollector", "PDFCollector", "GitHubCollector", "YouTubeCollector"]
