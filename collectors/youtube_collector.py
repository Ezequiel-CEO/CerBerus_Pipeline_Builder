#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coletor de conteúdo do YouTube para o PipelineBuilder.

Extrai transcrições e metadados de vídeos do YouTube.
"""

import os
import re
import hashlib
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import requests
from urllib.parse import urlparse, parse_qs

from cerberus_api.pipeline_builder.models import (
    DataSource,
    CollectionTask,
    ContentChunk,
    SourceType,
    ContentType,
    TaskStatus,
)
from cerberus_api.pipeline_builder.utils.storage import save_content_chunk
from cerberus_api.utils.logging_config import APILogger

# Configurar logger
logger = APILogger("youtube_collector")


class YouTubeCollector:
    """
    Coletor para extração de conteúdo de vídeos do YouTube.
    """

    def __init__(
        self, storage_dir: str = "data/youtube_content", api_key: Optional[str] = None
    ):
        """
        Inicializa o coletor YouTube.

        Args:
            storage_dir: Diretório para armazenamento temporário de conteúdo.
            api_key: Chave da API do YouTube (opcional).
        """
        self.storage_dir = storage_dir
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY")

        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"YouTubeCollector inicializado. Storage: {storage_dir}")

    def collect(self, task: CollectionTask, source: DataSource) -> List[ContentChunk]:
        """
        Coleta conteúdo de um vídeo ou canal do YouTube.

        Args:
            task: Tarefa de coleta.
            source: Fonte de dados.

        Returns:
            Lista de chunks de conteúdo coletados.
        """
        logger.info(f"Iniciando coleta do YouTube: {source.location}")

        if source.source_type != SourceType.YOUTUBE:
            logger.error(f"Tipo de fonte inválido: {source.source_type}")
            raise ValueError(
                f"YouTubeCollector só pode processar fontes do tipo YOUTUBE, recebido: {source.source_type}"
            )

        # Atualizar status da tarefa
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        chunks = []

        try:
            # Criar diretório para armazenar o conteúdo extraído
            output_path = os.path.join(
                self.storage_dir,
                f"{source.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            os.makedirs(output_path, exist_ok=True)

            # Verificar tipo de URL do YouTube
            youtube_id = self._extract_youtube_id(source.location)

            if not youtube_id:
                logger.error(
                    f"Não foi possível extrair ID do YouTube da URL: {source.location}"
                )
                raise ValueError(f"URL do YouTube inválida: {source.location}")

            # Extrair conteúdo do vídeo
            video_chunks = self._process_video(youtube_id, task.id, output_path)
            chunks.extend(video_chunks)

            # Atualizar metadados da tarefa
            task.metadata.update(
                {
                    "chunks_collected": len(chunks),
                    "total_content_length": sum(chunk.length for chunk in chunks),
                    "youtube_id": youtube_id,
                }
            )

            # Atualizar status da tarefa
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.content_path = output_path

            logger.info(
                f"Coleta concluída. {len(chunks)} chunks coletados do YouTube: {source.location}"
            )
            return chunks

        except Exception as e:
            logger.error(f"Erro durante coleta do YouTube {source.location}: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            raise

    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """
        Extrai o ID do vídeo ou canal do YouTube a partir da URL.

        Args:
            url: URL do YouTube.

        Returns:
            ID do vídeo ou canal, ou None se não for possível extrair.
        """
        # Padrões de URL do YouTube:
        # - youtube.com/watch?v=VIDEO_ID
        # - youtu.be/VIDEO_ID
        # - youtube.com/channel/CHANNEL_ID
        # - youtube.com/user/USERNAME

        if "youtube.com/watch" in url or "youtu.be/" in url:
            # É um vídeo
            if "youtube.com/watch" in url:
                parsed_url = urlparse(url)
                video_id = parse_qs(parsed_url.query).get("v", [None])[0]
                return video_id
            elif "youtu.be/" in url:
                return url.split("youtu.be/")[1].split("?")[0]

        elif "youtube.com/channel/" in url:
            # É um canal
            return url.split("youtube.com/channel/")[1].split("?")[0].split("/")[0]

        elif "youtube.com/user/" in url:
            # É um usuário
            return url.split("youtube.com/user/")[1].split("?")[0].split("/")[0]

        # Caso não identifique o padrão
        logger.warning(f"Formato de URL do YouTube não reconhecido: {url}")
        return None

    def _process_video(
        self, video_id: str, task_id: str, output_path: str
    ) -> List[ContentChunk]:
        """
        Processa um vídeo do YouTube e extrai seu conteúdo.

        Args:
            video_id: ID do vídeo do YouTube.
            task_id: ID da tarefa de coleta.
            output_path: Diretório para salvar chunks.

        Returns:
            Lista de chunks de conteúdo extraídos.
        """
        logger.info(f"Processando vídeo do YouTube: {video_id}")
        chunks = []

        try:
            # Obter dados do vídeo via API do YouTube
            video_data = self._get_video_data(video_id)

            # Se não tiver chave de API, criar um chunk simulado
            if not self.api_key:
                logger.warning("Sem chave de API do YouTube. Criando chunk simulado.")

                # Criar chunk com metadados básicos
                metadata_text = f"# Vídeo do YouTube\n\n"
                metadata_text += f"ID do vídeo: {video_id}\n"
                metadata_text += f"URL: https://www.youtube.com/watch?v={video_id}\n"
                metadata_text += (
                    f"Nota: Transcrição não disponível sem chave de API do YouTube.\n"
                )
                metadata_text += f"Para obter transcrições reais, configure uma chave de API do YouTube.\n"

                chunk_id = hashlib.md5(
                    f"youtube_{video_id}_metadata_{datetime.now().isoformat()}".encode()
                ).hexdigest()

                metadata_chunk = ContentChunk(
                    id=chunk_id,
                    task_id=task_id,
                    content_type=ContentType.TEXT,
                    text=metadata_text,
                    source_location=f"https://www.youtube.com/watch?v={video_id}",
                    metadata={
                        "video_id": video_id,
                        "content_type": "metadata",
                        "note": "Simulado (sem chave de API)",
                    },
                    length=len(metadata_text),
                    created_at=datetime.now(),
                )
                chunks.append(metadata_chunk)
                save_content_chunk(
                    metadata_chunk, os.path.join(output_path, "metadata.json")
                )

            else:
                # Criar chunk com metadados do vídeo
                metadata_text = f"# {video_data.get('title', 'Vídeo do YouTube')}\n\n"
                metadata_text += f"ID do vídeo: {video_id}\n"
                metadata_text += f"Canal: {video_data.get('channel_title', 'N/A')}\n"
                metadata_text += (
                    f"Data de publicação: {video_data.get('published_at', 'N/A')}\n"
                )
                metadata_text += f"Duração: {video_data.get('duration', 'N/A')}\n"
                metadata_text += (
                    f"Visualizações: {video_data.get('view_count', 'N/A')}\n"
                )
                metadata_text += f"Likes: {video_data.get('like_count', 'N/A')}\n"
                metadata_text += (
                    f"Descrição:\n\n{video_data.get('description', 'N/A')}\n"
                )

                chunk_id = hashlib.md5(
                    f"youtube_{video_id}_metadata_{datetime.now().isoformat()}".encode()
                ).hexdigest()

                metadata_chunk = ContentChunk(
                    id=chunk_id,
                    task_id=task_id,
                    content_type=ContentType.TEXT,
                    text=metadata_text,
                    source_location=f"https://www.youtube.com/watch?v={video_id}",
                    metadata={
                        "video_id": video_id,
                        "content_type": "metadata",
                        "video_data": video_data,
                    },
                    length=len(metadata_text),
                    created_at=datetime.now(),
                )
                chunks.append(metadata_chunk)
                save_content_chunk(
                    metadata_chunk, os.path.join(output_path, "metadata.json")
                )

                # Obter transcrição do vídeo
                transcript = self._get_video_transcript(video_id)

                if transcript:
                    # Criar chunk com a transcrição
                    transcript_text = f"# Transcrição: {video_data.get('title', 'Vídeo do YouTube')}\n\n"
                    transcript_text += transcript

                    chunk_id = hashlib.md5(
                        f"youtube_{video_id}_transcript_{datetime.now().isoformat()}".encode()
                    ).hexdigest()

                    transcript_chunk = ContentChunk(
                        id=chunk_id,
                        task_id=task_id,
                        content_type=ContentType.TEXT,
                        text=transcript_text,
                        source_location=f"https://www.youtube.com/watch?v={video_id}",
                        metadata={
                            "video_id": video_id,
                            "content_type": "transcript",
                            "video_title": video_data.get("title", "Vídeo do YouTube"),
                        },
                        length=len(transcript_text),
                        created_at=datetime.now(),
                    )
                    chunks.append(transcript_chunk)
                    save_content_chunk(
                        transcript_chunk, os.path.join(output_path, "transcript.json")
                    )

            logger.info(
                f"Processamento do vídeo concluído. {len(chunks)} chunks extraídos."
            )
            return chunks

        except Exception as e:
            logger.error(f"Erro ao processar vídeo do YouTube {video_id}: {e}")
            # Criar um chunk de erro
            error_chunk = ContentChunk(
                id=hashlib.md5(
                    f"youtube_{video_id}_error_{datetime.now().isoformat()}".encode()
                ).hexdigest(),
                task_id=task_id,
                content_type=ContentType.TEXT,
                text=f"Erro ao processar vídeo do YouTube {video_id}: {str(e)}",
                source_location=f"https://www.youtube.com/watch?v={video_id}",
                metadata={"error": str(e), "video_id": video_id},
                length=0,
                created_at=datetime.now(),
            )
            chunks.append(error_chunk)
            save_content_chunk(error_chunk, os.path.join(output_path, "error.json"))
            return chunks

    def _get_video_data(self, video_id: str) -> Dict[str, Any]:
        """
        Obtém dados de um vídeo do YouTube via API.

        Args:
            video_id: ID do vídeo.

        Returns:
            Dicionário com dados do vídeo.
        """
        # Se não tiver chave de API, retornar dados básicos
        if not self.api_key:
            return {
                "title": f"Vídeo {video_id}",
                "channel_title": "Canal desconhecido",
                "published_at": "Desconhecido",
                "description": "Descrição não disponível sem chave de API.",
            }

        try:
            # Montar URL da API
            api_url = f"https://www.googleapis.com/youtube/v3/videos"
            params = {
                "part": "snippet,contentDetails,statistics",
                "id": video_id,
                "key": self.api_key,
            }

            response = requests.get(api_url, params=params)
            response.raise_for_status()

            data = response.json()

            if not data.get("items"):
                logger.warning(f"Nenhum dado encontrado para o vídeo {video_id}")
                return {}

            video_item = data["items"][0]
            snippet = video_item.get("snippet", {})
            content_details = video_item.get("contentDetails", {})
            statistics = video_item.get("statistics", {})

            return {
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "published_at": snippet.get("publishedAt", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "channel_id": snippet.get("channelId", ""),
                "tags": snippet.get("tags", []),
                "category_id": snippet.get("categoryId", ""),
                "duration": content_details.get("duration", ""),
                "view_count": statistics.get("viewCount", 0),
                "like_count": statistics.get("likeCount", 0),
                "comment_count": statistics.get("commentCount", 0),
            }

        except Exception as e:
            logger.error(f"Erro ao obter dados do vídeo {video_id}: {e}")
            return {}

    def _get_video_transcript(self, video_id: str) -> Optional[str]:
        """
        Obtém a transcrição de um vídeo do YouTube.

        Args:
            video_id: ID do vídeo.

        Returns:
            Texto da transcrição ou None se não disponível.
        """
        # Método simplificado - em uma implementação real, usaríamos
        # bibliotecas como youtube-transcript-api ou a API oficial

        # Aqui, retornamos uma mensagem informativa
        if not self.api_key:
            return None

        try:
            # Na implementação real, aqui seria o código para obter a transcrição
            # via API do YouTube ou youtube-transcript-api

            # Para este exemplo, retornamos uma mensagem simulada
            return (
                "Esta é uma transcrição simulada para demonstração do PipelineBuilder.\n\n"
                f"Em uma implementação real, aqui estaria a transcrição completa do vídeo {video_id}.\n"
                "A transcrição incluiria todo o conteúdo falado e possivelmente legendas.\n\n"
                "Para implementar corretamente, é necessário utilizar bibliotecas como youtube-transcript-api\n"
                "ou a API oficial do YouTube Data API v3 com autenticação apropriada."
            )

        except Exception as e:
            logger.error(f"Erro ao obter transcrição do vídeo {video_id}: {e}")
            return None
