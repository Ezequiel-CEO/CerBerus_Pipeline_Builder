#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coletor de conteúdo web para o PipelineBuilder.

Utiliza BeautifulSoup e requests para extrair conteúdo textual
de páginas web, blogs e artigos online.
"""

import os
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from urllib.parse import urlparse, urljoin
import hashlib
import re

import requests
from bs4 import BeautifulSoup
import trafilatura
try:
    from readability import Document
    Article = Document  # Compatibilidade com versões antigas
except ImportError:
    try:
        from readability import Article
    except ImportError:
        Article = None  # Fallback se não conseguir importar
import justext

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
logger = APILogger("web_collector")

# Headers para simular um navegador
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}

# Lista de tags geralmente irrelevantes
NOISY_TAGS = [
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "header",
    "footer",
    "nav",
    "aside",
    "form",
    "button",
    "input",
]

# Padrões de URL para ignorar
IGNORE_URL_PATTERNS = [
    r"\.(?:jpg|jpeg|png|gif|bmp|svg|webp|css|js|woff|woff2|ttf|eot)$",
    r"/(?:login|signup|register|cart|checkout|privacy|terms)",
    r"(?:facebook\.com|twitter\.com|instagram\.com|linkedin\.com)/share",
]


class WebCollector:
    """
    Coletor para extração de conteúdo web.
    """

    def __init__(self, storage_dir: str = "data/web_content"):
        """
        Inicializa o coletor web.

        Args:
            storage_dir: Diretório para armazenamento temporário de conteúdo.
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def collect(self, task: CollectionTask, source: DataSource) -> List[ContentChunk]:
        """
        Coleta conteúdo de uma fonte web.

        Args:
            task: Tarefa de coleta.
            source: Fonte de dados.

        Returns:
            Lista de chunks de conteúdo coletados.
        """
        logger.info(f"Iniciando coleta da URL: {source.location}")

        if source.source_type != SourceType.WEB:
            logger.error(f"Tipo de fonte inválido: {source.source_type}")
            raise ValueError(
                f"WebCollector só pode processar fontes do tipo WEB, recebido: {source.source_type}"
            )

        # Atualizar status da tarefa
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        chunks = []
        visited_urls = set()

        try:
            # Coletar a página principal
            main_chunks = self._process_url(source.location, task.id, is_main_page=True)
            chunks.extend(main_chunks)
            visited_urls.add(source.location)

            # Se houver instruções para seguir links, processar páginas secundárias
            if source.metadata.get("follow_links", False):
                max_pages = source.metadata.get("max_pages", 5)
                depth = source.metadata.get("depth", 1)

                links = self._extract_links(source.location)
                valid_links = self._filter_links(links, source.location, visited_urls)

                # Limitar ao número máximo de páginas
                valid_links = valid_links[:max_pages]

                for link in valid_links:
                    if len(visited_urls) >= max_pages:
                        break

                    logger.info(f"Processando link secundário: {link}")
                    secondary_chunks = self._process_url(
                        link, task.id, is_main_page=False
                    )
                    chunks.extend(secondary_chunks)
                    visited_urls.add(link)

                    # Respeitar limites de requisição
                    time.sleep(1)

            # Salvar metadados da coleta
            task.metadata.update(
                {
                    "pages_collected": len(visited_urls),
                    "chunks_collected": len(chunks),
                    "total_content_length": sum(chunk.length for chunk in chunks),
                }
            )

            # Atualizar status da tarefa
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            # Criar o caminho para o conteúdo coletado
            output_path = os.path.join(
                self.storage_dir,
                f"{source.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            os.makedirs(output_path, exist_ok=True)

            # Salvar os chunks coletados
            for i, chunk in enumerate(chunks):
                save_content_chunk(chunk, os.path.join(output_path, f"chunk_{i}.json"))

            # Salvar metadados da coleta
            with open(
                os.path.join(output_path, "metadata.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "task_id": task.id,
                        "source_id": source.id,
                        "source_name": source.name,
                        "source_location": source.location,
                        "collected_at": datetime.now().isoformat(),
                        "pages_collected": len(visited_urls),
                        "chunks_collected": len(chunks),
                        "visited_urls": list(visited_urls),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            task.content_path = output_path

            logger.info(
                f"Coleta concluída. {len(chunks)} chunks coletados de {len(visited_urls)} páginas."
            )
            return chunks

        except Exception as e:
            logger.error(f"Erro durante coleta: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            raise

    def _process_url(
        self, url: str, task_id: str, is_main_page: bool = False
    ) -> List[ContentChunk]:
        """
        Processa uma URL e extrai o conteúdo.

        Args:
            url: URL para processar.
            task_id: ID da tarefa de coleta.
            is_main_page: Se é a página principal da fonte.

        Returns:
            Lista de chunks de conteúdo extraídos.
        """
        logger.info(f"Processando URL: {url}")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            html_content = response.text

            # Usar diferentes métodos de extração para maior eficácia
            chunks = []

            # 1. Trafilatura (bom para artigos e blogs)
            trafilatura_text = trafilatura.extract(
                html_content, include_comments=False, include_tables=True
            )
            if trafilatura_text and len(trafilatura_text.strip()) > 100:
                chunks.append(
                    self._create_content_chunk(
                        task_id=task_id,
                        content=trafilatura_text,
                        source_location=url,
                        extractor="trafilatura",
                        is_main_content=True,
                    )
                )

            # 2. Readability (alternativa para extração de artigos)
            doc = Article(html_content)
            readability_title = doc.title
            readability_content = doc.summary

            if readability_content:
                # Limpar tags HTML
                soup = BeautifulSoup(readability_content, "html.parser")
                readability_text = soup.get_text(separator="\n\n")

                if readability_text and len(readability_text.strip()) > 100:
                    # Evitar duplicação se for muito similar ao resultado do Trafilatura
                    if (
                        not trafilatura_text
                        or self._content_similarity(trafilatura_text, readability_text)
                        < 0.7
                    ):
                        chunks.append(
                            self._create_content_chunk(
                                task_id=task_id,
                                content=readability_text,
                                source_location=url,
                                extractor="readability",
                                is_main_content=True,
                            )
                        )

            # 3. JusText (bom para conteúdo principal de páginas)
            try:
                paragraphs = justext.justext(
                    response.content, justext.get_stoplist("english")
                )
                justext_content = "\n\n".join(
                    [p.text for p in paragraphs if not p.is_boilerplate]
                )

                if justext_content and len(justext_content.strip()) > 100:
                    # Evitar duplicação se for muito similar aos resultados anteriores
                    if (
                        not trafilatura_text
                        or self._content_similarity(trafilatura_text, justext_content)
                        < 0.7
                    ) and (
                        not readability_text
                        or self._content_similarity(readability_text, justext_content)
                        < 0.7
                    ):
                        chunks.append(
                            self._create_content_chunk(
                                task_id=task_id,
                                content=justext_content,
                                source_location=url,
                                extractor="justext",
                                is_main_content=True,
                            )
                        )
            except Exception as e:
                logger.warning(f"Erro no JusText: {e}")

            # 4. BeautifulSoup para extração direta
            soup = BeautifulSoup(html_content, "html.parser")

            # Remover elementos irrelevantes
            for tag in NOISY_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()

            # Extrair blocos de código, se presentes
            code_blocks = soup.find_all(["pre", "code"])
            for i, code in enumerate(code_blocks):
                if code.text and len(code.text.strip()) > 50:
                    chunks.append(
                        self._create_content_chunk(
                            task_id=task_id,
                            content=code.text,
                            source_location=url,
                            extractor="beautifulsoup",
                            is_main_content=False,
                            content_type=ContentType.CODE,
                            metadata={"code_block_index": i},
                        )
                    )

            # Se ainda não temos conteúdo suficiente, extrair parágrafos
            if len(chunks) < 2:
                paragraphs = soup.find_all(
                    ["p", "article", "section", "div.content", "div.article"]
                )
                filtered_paragraphs = []

                for p in paragraphs:
                    text = p.get_text(separator=" ").strip()
                    # Filtrar parágrafos muito curtos ou vazios
                    if len(text) > 100 and not self._is_navigation_or_boilerplate(p):
                        filtered_paragraphs.append(text)

                if filtered_paragraphs:
                    combined_text = "\n\n".join(filtered_paragraphs)
                    chunks.append(
                        self._create_content_chunk(
                            task_id=task_id,
                            content=combined_text,
                            source_location=url,
                            extractor="beautifulsoup_paragraphs",
                            is_main_content=True,
                        )
                    )

            # Extrair metadados da página (título, descrição, etc.)
            metadata = {
                "title": soup.title.string if soup.title else None,
                "description": (
                    soup.find("meta", attrs={"name": "description"})["content"]
                    if soup.find("meta", attrs={"name": "description"})
                    else None
                ),
                "keywords": (
                    soup.find("meta", attrs={"name": "keywords"})["content"]
                    if soup.find("meta", attrs={"name": "keywords"})
                    else None
                ),
            }

            # Adicionar metadados aos chunks
            for chunk in chunks:
                chunk.metadata.update({"page_metadata": metadata})

            return chunks

        except Exception as e:
            logger.error(f"Erro ao processar URL {url}: {e}")
            # Criar um chunk vazio para registrar o erro
            error_chunk = ContentChunk(
                id=str(
                    hashlib.md5(
                        f"{url}_{datetime.now().isoformat()}".encode()
                    ).hexdigest()
                ),
                task_id=task_id,
                content_type=ContentType.TEXT,
                text=f"Erro ao processar: {str(e)}",
                source_location=url,
                metadata={"error": str(e), "error_type": type(e).__name__},
                length=0,
                created_at=datetime.now(),
            )
            return [error_chunk]

    def _extract_links(self, url: str) -> List[str]:
        """
        Extrai links de uma página web.

        Args:
            url: URL da página.

        Returns:
            Lista de URLs encontradas na página.
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            base_url = response.url

            links = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()

                # Ignorar links vazios ou ancoras
                if not href or href.startswith("#"):
                    continue

                # Construir URL absoluta
                absolute_url = urljoin(base_url, href)

                # Garantir que estamos no mesmo domínio
                if self._is_same_domain(base_url, absolute_url):
                    links.append(absolute_url)

            return links

        except Exception as e:
            logger.error(f"Erro ao extrair links de {url}: {e}")
            return []

    def _filter_links(
        self, links: List[str], base_url: str, visited_urls: set
    ) -> List[str]:
        """
        Filtra links por relevância e remove duplicatas.

        Args:
            links: Lista de links para filtrar.
            base_url: URL base para comparação.
            visited_urls: Conjunto de URLs já visitadas.

        Returns:
            Lista filtrada de links.
        """
        filtered_links = []
        base_domain = urlparse(base_url).netloc

        for link in links:
            # Ignorar se já visitado
            if link in visited_urls:
                continue

            # Verificar se está no mesmo domínio
            link_domain = urlparse(link).netloc
            if not link_domain.endswith(base_domain) and not base_domain.endswith(
                link_domain
            ):
                continue

            # Verificar padrões de URL a ignorar
            skip = False
            for pattern in IGNORE_URL_PATTERNS:
                if re.search(pattern, link, re.IGNORECASE):
                    skip = True
                    break

            if not skip:
                filtered_links.append(link)

        return filtered_links

    def _is_same_domain(self, url1: str, url2: str) -> bool:
        """
        Verifica se duas URLs são do mesmo domínio.

        Args:
            url1: Primeira URL.
            url2: Segunda URL.

        Returns:
            True se as URLs forem do mesmo domínio.
        """
        domain1 = urlparse(url1).netloc
        domain2 = urlparse(url2).netloc

        # Considerar subdomínios
        return (
            domain1 == domain2
            or domain1.endswith(f".{domain2}")
            or domain2.endswith(f".{domain1}")
        )

    def _is_navigation_or_boilerplate(self, element) -> bool:
        """
        Verifica se um elemento HTML é provavelmente parte da navegação ou boilerplate.

        Args:
            element: Elemento HTML para verificar.

        Returns:
            True se o elemento for provavelmente navegação ou boilerplate.
        """
        # Verificar classes e IDs comuns de navegação
        nav_patterns = [
            "nav",
            "menu",
            "header",
            "footer",
            "sidebar",
            "comment",
            "social",
        ]
        classes = element.get("class", [])
        element_id = element.get("id", "")

        # Converter classes para string para facilitar verificação
        class_str = (
            " ".join(classes).lower()
            if isinstance(classes, list)
            else str(classes).lower()
        )
        id_str = str(element_id).lower()

        for pattern in nav_patterns:
            if (pattern in class_str) or (pattern in id_str):
                return True

        # Verificar se é muito curto ou tem muitos links
        text = element.get_text().strip()
        links = element.find_all("a")

        if len(text) < 100 and len(links) > 2:
            # Provavelmente um menu
            return True

        return False

    def _create_content_chunk(
        self,
        task_id: str,
        content: str,
        source_location: str,
        extractor: str,
        is_main_content: bool,
        content_type: ContentType = ContentType.TEXT,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContentChunk:
        """
        Cria um chunk de conteúdo.

        Args:
            task_id: ID da tarefa de coleta.
            content: Conteúdo textual.
            source_location: URL da fonte.
            extractor: Nome do extrator utilizado.
            is_main_content: Se é o conteúdo principal da página.
            content_type: Tipo de conteúdo.
            metadata: Metadados adicionais.

        Returns:
            Chunk de conteúdo criado.
        """
        chunk_id = hashlib.md5(
            f"{source_location}_{extractor}_{datetime.now().isoformat()}".encode()
        ).hexdigest()

        metadata = metadata or {}
        metadata.update(
            {
                "extractor": extractor,
                "is_main_content": is_main_content,
                "extracted_at": datetime.now().isoformat(),
            }
        )

        return ContentChunk(
            id=chunk_id,
            task_id=task_id,
            content_type=content_type,
            text=content,
            source_location=source_location,
            metadata=metadata,
            length=len(content),
            created_at=datetime.now(),
        )

    def _content_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula uma medida de similaridade simples entre dois textos.

        Args:
            text1: Primeiro texto.
            text2: Segundo texto.

        Returns:
            Valor de similaridade entre 0 e 1.
        """
        # Implementação simples de similaridade baseada em comprimento
        len1 = len(text1)
        len2 = len(text2)

        # Se um dos textos for muito pequeno, retornar baixa similaridade
        if len1 < 100 or len2 < 100:
            return 0.0

        # Calcular Jaccard similarity em parágrafos
        paragraphs1 = set(p.strip() for p in text1.split("\n") if len(p.strip()) > 50)
        paragraphs2 = set(p.strip() for p in text2.split("\n") if len(p.strip()) > 50)

        if not paragraphs1 or not paragraphs2:
            return 0.0

        intersection = len(paragraphs1.intersection(paragraphs2))
        union = len(paragraphs1.union(paragraphs2))

        return intersection / union if union > 0 else 0.0
