#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modelos de dados para o PipelineBuilder.

Define as estruturas de dados utilizadas para representar
fontes de dados, tarefas de coleta, resultados de análise
e pipelines de treinamento.
"""

import enum
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator, HttpUrl
import uuid


class SourceType(str, enum.Enum):
    """Tipos de fontes de dados suportados."""

    WEB = "web"  # Páginas web, artigos, blogs
    GITHUB = "github"  # Repositórios e arquivos do GitHub
    PDF = "pdf"  # Documentos PDF
    YOUTUBE = "youtube"  # Vídeos do YouTube
    AUDIO = "audio"  # Arquivos de áudio
    TEXT = "text"  # Arquivos de texto plano
    CUSTOM = "custom"  # Fonte personalizada


class ContentType(str, enum.Enum):
    """Tipos de conteúdo extraídos."""

    TEXT = "text"  # Texto extraído
    CODE = "code"  # Código-fonte
    STRUCTURED = "structured"  # Dados estruturados (tabelas, etc)
    MIXED = "mixed"  # Conteúdo misto
    MARKDOWN = "markdown"  # Conteúdo em formato Markdown
    HTML = "html"  # Conteúdo em formato HTML


class ContentCategory(str, enum.Enum):
    """Categorias de conteúdo para classificação."""

    TECHNOLOGY = "technology"
    SCIENCE = "science"
    SECURITY = "security"
    PROGRAMMING = "programming"
    TUTORIALS = "tutorials"
    DOCUMENTATION = "documentation"
    REFERENCE = "reference"
    ACADEMIC = "academic"
    OTHER = "other"


class TaskStatus(str, enum.Enum):
    """Status de uma tarefa de coleta ou processamento."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class PipelineStatus(str, enum.Enum):
    """Status de um pipeline de treinamento."""

    DRAFT = "draft"  # Pipeline em construção
    READY = "ready"  # Pronto para aprovação
    APPROVED = "approved"  # Aprovado pelo usuário
    REJECTED = "rejected"  # Rejeitado pelo usuário
    TRAINING = "training"  # Em treinamento
    COMPLETED = "completed"  # Treinamento concluído
    FAILED = "failed"  # Falha no treinamento


class DataSource(BaseModel):
    """
    Representa uma fonte de dados para coleta.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Nome da fonte de dados")
    source_type: SourceType = Field(..., description="Tipo da fonte de dados")
    location: str = Field(..., description="URL, caminho ou localização da fonte")
    description: Optional[str] = Field(None, description="Descrição da fonte")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadados adicionais"
    )
    created_at: datetime = Field(default_factory=datetime.now)

    @validator("location")
    def validate_location(cls, v, values):
        """Valida a localização com base no tipo de fonte."""
        source_type = values.get("source_type")

        if source_type == SourceType.WEB:
            # Validação básica de URL
            if not (v.startswith("http://") or v.startswith("https://")):
                raise ValueError("URL da web deve começar com http:// ou https://")

        elif source_type == SourceType.GITHUB:
            # Validação básica de URL do GitHub
            if not (
                "github.com" in v
                and (v.startswith("http://") or v.startswith("https://"))
            ):
                raise ValueError("URL do GitHub inválida")

        return v


class CollectionTask(BaseModel):
    """
    Representa uma tarefa de coleta de dados de uma fonte.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = Field(..., description="ID da fonte de dados")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    error_message: Optional[str] = Field(None)
    content_path: Optional[str] = Field(
        None, description="Caminho para o conteúdo coletado"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("started_at")
    def validate_started_at(cls, v, values):
        """Valida que started_at seja posterior a created_at."""
        if v and values.get("created_at") and v < values.get("created_at"):
            raise ValueError("started_at deve ser posterior a created_at")
        return v

    @validator("completed_at")
    def validate_completed_at(cls, v, values):
        """Valida que completed_at seja posterior a started_at."""
        if v and values.get("started_at") and v < values.get("started_at"):
            raise ValueError("completed_at deve ser posterior a started_at")
        return v


class ContentChunk(BaseModel):
    """
    Representa um pedaço de conteúdo processado.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = Field(..., description="ID da tarefa de coleta")
    content_type: ContentType = Field(...)
    text: str = Field(..., description="Conteúdo textual")
    source_location: str = Field(
        ..., description="Localização específica na fonte (URL, página, etc)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    length: int = Field(..., description="Comprimento do texto em caracteres")
    created_at: datetime = Field(default_factory=datetime.now)


class AnalysisResult(BaseModel):
    """
    Resultado da análise de um chunk de conteúdo.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = Field(..., description="ID da tarefa de análise")
    chunk_id: str = Field(..., description="ID do chunk analisado")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class AnalysisTask(BaseModel):
    """
    Representa uma tarefa de análise de conteúdo.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_ids: List[str] = Field(..., description="IDs dos chunks a serem analisados")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    error_message: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("started_at")
    def validate_started_at(cls, v, values):
        """Valida que started_at seja posterior a created_at."""
        if v and values.get("created_at") and v < values.get("created_at"):
            raise ValueError("started_at deve ser posterior a created_at")
        return v

    @validator("completed_at")
    def validate_completed_at(cls, v, values):
        """Valida que completed_at seja posterior a started_at."""
        if v and values.get("started_at") and v < values.get("started_at"):
            raise ValueError("completed_at deve ser posterior a started_at")
        return v


class ProcessingTask(BaseModel):
    """
    Representa uma tarefa de processamento de pipeline.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_id: str = Field(..., description="ID do pipeline")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    error_message: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("started_at")
    def validate_started_at(cls, v, values):
        """Valida que started_at seja posterior a created_at."""
        if v and values.get("created_at") and v < values.get("created_at"):
            raise ValueError("started_at deve ser posterior a created_at")
        return v

    @validator("completed_at")
    def validate_completed_at(cls, v, values):
        """Valida que completed_at seja posterior a started_at."""
        if v and values.get("started_at") and v < values.get("started_at"):
            raise ValueError("completed_at deve ser posterior a started_at")
        return v


class ProcessingResult(BaseModel):
    """
    Resultado do processamento de um pipeline.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = Field(..., description="ID da tarefa de processamento")
    pipeline_id: str = Field(..., description="ID do pipeline")
    chunk_id: str = Field(..., description="ID do chunk processado")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class PipelineStage(BaseModel):
    """
    Representa um estágio em um pipeline.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Nome do estágio")
    description: Optional[str] = Field(None, description="Descrição do estágio")
    steps: List["PipelineStep"] = Field(
        default_factory=list, description="Passos do estágio"
    )


class PipelineStep(BaseModel):
    """
    Representa um passo em um estágio do pipeline.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Nome do passo")
    description: Optional[str] = Field(None, description="Descrição do passo")
    processor_type: str = Field(..., description="Tipo de processador")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parâmetros do processador"
    )
    source_ids: List[str] = Field(
        default_factory=list, description="IDs das fontes utilizadas"
    )


class Pipeline(BaseModel):
    """
    Representa um pipeline de treinamento.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Nome do pipeline")
    description: str = Field(..., description="Descrição do pipeline")
    status: PipelineStatus = Field(default=PipelineStatus.DRAFT)
    stages: List[PipelineStage] = Field(
        default_factory=list, description="Estágios do pipeline"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @validator("updated_at")
    def validate_updated_at(cls, v, values):
        """Valida que updated_at seja posterior ou igual a created_at."""
        if v and values.get("created_at") and v < values.get("created_at"):
            raise ValueError("updated_at deve ser posterior ou igual a created_at")
        return v


class PipelineApproval(BaseModel):
    """
    Registro de aprovação/rejeição de um pipeline.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_id: str = Field(..., description="ID do pipeline")
    approved: bool = Field(..., description="Se o pipeline foi aprovado")
    user_id: str = Field(..., description="ID do usuário que aprovou/rejeitou")
    comment: Optional[str] = Field(None, description="Comentário opcional")
    created_at: datetime = Field(default_factory=datetime.now)
