#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline Builder: Ferramenta para construção e gerenciamento de pipelines de NLP/AI
"""

# Versão do pacote
__version__ = "0.6.0"

# Imports básicos
from . import ai_manager
from . import utils
from . import collectors
from . import processors
from . import storage
from . import analyzers

# Exportar componentes principais
__all__ = ["ai_manager", "utils", "collectors", "processors", "storage", "analyzers"]

# Funções de alto nível se houver
from .core import create_pipeline, run_pipeline, load_model, save_model

# Adicionar funções ao __all__
__all__ += ["create_pipeline", "run_pipeline", "load_model", "save_model"]

from cerberus_api.pipeline_builder.core import PipelineBuilder
from cerberus_api.pipeline_builder.models import (
    Pipeline,
    DataSource,
    CollectionTask,
    AnalysisResult,
    SourceType,
    ContentType,
    ContentCategory,
    PipelineStatus,
)
from cerberus_api.pipeline_builder.ui import PipelineDashboard

__all__ += [
    "PipelineBuilder",
    "Pipeline",
    "DataSource",
    "CollectionTask",
    "AnalysisResult",
    "SourceType",
    "ContentType",
    "ContentCategory",
    "PipelineStatus",
    "PipelineDashboard",
]
