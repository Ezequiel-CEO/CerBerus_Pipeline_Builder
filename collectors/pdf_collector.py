#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coletor de conteúdo PDF para o PipelineBuilder.

Utiliza PyPDF2 e outras bibliotecas para extrair texto
e metadados de arquivos PDF.
"""

import os
import re
import hashlib
import tempfile
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from urllib.parse import urlparse
from urllib.request import urlretrieve
import shutil

import PyPDF2
import fitz  # PyMuPDF
import requests
from pdfminer.high_level import extract_text as pdfminer_extract_text

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
logger = APILogger("pdf_collector")


class PDFCollector:
    """
    Coletor para extração de conteúdo de arquivos PDF.
    """

    def __init__(self, storage_dir: str = "data/pdf_content"):
        """
        Inicializa o coletor PDF.

        Args:
            storage_dir: Diretório para armazenamento temporário de conteúdo.
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"PDFCollector inicializado. Storage: {storage_dir}")

    def collect(self, task: CollectionTask, source: DataSource) -> List[ContentChunk]:
        """
        Coleta conteúdo de uma fonte PDF.

        Args:
            task: Tarefa de coleta.
            source: Fonte de dados.

        Returns:
            Lista de chunks de conteúdo coletados.
        """
        logger.info(f"Iniciando coleta do PDF: {source.location}")

        if source.source_type != SourceType.PDF:
            logger.error(f"Tipo de fonte inválido: {source.source_type}")
            raise ValueError(
                f"PDFCollector só pode processar fontes do tipo PDF, recebido: {source.source_type}"
            )

        # Atualizar status da tarefa
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        chunks = []
        pdf_path = None

        try:
            # Criar diretório para armazenar o conteúdo extraído
            output_path = os.path.join(
                self.storage_dir,
                f"{source.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            os.makedirs(output_path, exist_ok=True)

            # Verificar se é uma URL ou um caminho local
            if source.location.startswith("http://") or source.location.startswith(
                "https://"
            ):
                # Se for URL, baixar o PDF
                pdf_path = self._download_pdf(source.location, output_path)
            else:
                # Se for caminho local, usar diretamente
                pdf_path = source.location
                if not os.path.exists(pdf_path):
                    raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")

            # Extrair conteúdo do PDF
            chunks = self._process_pdf(pdf_path, task.id, output_path)

            # Atualizar metadados da tarefa
            task.metadata.update(
                {
                    "chunks_collected": len(chunks),
                    "total_content_length": sum(chunk.length for chunk in chunks),
                    "pdf_path": pdf_path,
                }
            )

            # Atualizar status da tarefa
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.content_path = output_path

            logger.info(
                f"Coleta concluída. {len(chunks)} chunks coletados do PDF: {source.location}"
            )
            return chunks

        except Exception as e:
            logger.error(f"Erro durante coleta do PDF {source.location}: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            raise

    def _download_pdf(self, url: str, output_dir: str) -> str:
        """
        Baixa um arquivo PDF de uma URL.

        Args:
            url: URL do arquivo PDF.
            output_dir: Diretório onde salvar o arquivo PDF.

        Returns:
            Caminho do arquivo PDF baixado.
        """
        logger.info(f"Baixando PDF de {url}")

        # Extrair nome do arquivo da URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        # Se não for possível determinar o nome, usar um hash da URL
        if not filename or not filename.lower().endswith(".pdf"):
            filename = f"{hashlib.md5(url.encode()).hexdigest()}.pdf"

        # Caminho completo para salvar o arquivo
        filepath = os.path.join(output_dir, filename)

        # Baixar o arquivo
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Verificar se o conteúdo é realmente um PDF
            content_type = response.headers.get("Content-Type", "")
            if "application/pdf" not in content_type and not url.lower().endswith(
                ".pdf"
            ):
                logger.warning(f"O conteúdo pode não ser um PDF: {content_type}")

            # Salvar o arquivo
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"PDF baixado e salvo em: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Erro ao baixar PDF de {url}: {e}")
            raise

    def _process_pdf(
        self, pdf_path: str, task_id: str, output_dir: str
    ) -> List[ContentChunk]:
        """
        Processa um arquivo PDF e extrai seu conteúdo.

        Args:
            pdf_path: Caminho do arquivo PDF.
            task_id: ID da tarefa de coleta.
            output_dir: Diretório para salvar chunks.

        Returns:
            Lista de chunks de conteúdo extraídos.
        """
        logger.info(f"Processando PDF: {pdf_path}")
        chunks = []

        try:
            # Extrair metadados do PDF com PyPDF2
            pdf_metadata = self._extract_pdf_metadata(pdf_path)

            # Tentar diferentes métodos de extração para obter o melhor resultado
            # 1. Extrair com PyMuPDF (fitz)
            pymupdf_chunks = self._extract_with_pymupdf(pdf_path, task_id, pdf_metadata)
            if pymupdf_chunks:
                chunks.extend(pymupdf_chunks)
                logger.info(f"Extraídos {len(pymupdf_chunks)} chunks com PyMuPDF")

            # 2. Se PyMuPDF falhar ou extrair pouco texto, tentar com PDFMiner
            if len(chunks) == 0 or sum(chunk.length for chunk in chunks) < 1000:
                pdfminer_chunks = self._extract_with_pdfminer(
                    pdf_path, task_id, pdf_metadata
                )
                if pdfminer_chunks:
                    # Se PyMuPDF já extraiu algo, verificar se PDFMiner trouxe conteúdo melhor
                    if len(chunks) > 0:
                        # Comparar tamanho total e usar o que extraiu mais conteúdo
                        pymupdf_size = sum(c.length for c in pymupdf_chunks)
                        pdfminer_size = sum(c.length for c in pdfminer_chunks)

                        if (
                            pdfminer_size > pymupdf_size * 1.2
                        ):  # Se PDFMiner extraiu pelo menos 20% mais
                            chunks = pdfminer_chunks
                            logger.info(
                                f"Substituído por {len(pdfminer_chunks)} chunks do PDFMiner (extraiu mais conteúdo)"
                            )
                    else:
                        chunks = pdfminer_chunks
                        logger.info(
                            f"Extraídos {len(pdfminer_chunks)} chunks com PDFMiner"
                        )

            # 3. Se ainda falhar, tentar com PyPDF2 como último recurso
            if len(chunks) == 0 or sum(chunk.length for chunk in chunks) < 1000:
                pypdf2_chunks = self._extract_with_pypdf2(
                    pdf_path, task_id, pdf_metadata
                )
                if pypdf2_chunks:
                    chunks = pypdf2_chunks
                    logger.info(f"Extraídos {len(pypdf2_chunks)} chunks com PyPDF2")

            # Salvar os chunks coletados
            for i, chunk in enumerate(chunks):
                save_content_chunk(chunk, os.path.join(output_dir, f"chunk_{i}.json"))

            logger.info(
                f"Processamento do PDF concluído. {len(chunks)} chunks extraídos."
            )
            return chunks

        except Exception as e:
            logger.error(f"Erro ao processar PDF {pdf_path}: {e}")
            # Criar um chunk de erro para registrar o problema
            error_chunk = ContentChunk(
                id=hashlib.md5(
                    f"{pdf_path}_{datetime.now().isoformat()}".encode()
                ).hexdigest(),
                task_id=task_id,
                content_type=ContentType.TEXT,
                text=f"Erro ao processar PDF: {str(e)}",
                source_location=pdf_path,
                metadata={"error": str(e), "error_type": type(e).__name__},
                length=0,
                created_at=datetime.now(),
            )
            chunks.append(error_chunk)
            return chunks

    def _extract_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extrai metadados de um arquivo PDF.

        Args:
            pdf_path: Caminho do arquivo PDF.

        Returns:
            Dicionário com metadados.
        """
        metadata = {}

        try:
            # Usar PyPDF2 para metadados
            with open(pdf_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)

                # Informações básicas
                metadata["page_count"] = len(pdf_reader.pages)

                # Metadados do documento
                pdf_info = pdf_reader.metadata
                if pdf_info:
                    for key in pdf_info:
                        clean_key = key.lower().replace("/", "_")
                        if isinstance(pdf_info[key], (str, int, float)):
                            metadata[clean_key] = pdf_info[key]

            # Tentar extrair metadados adicionais com PyMuPDF
            try:
                doc = fitz.open(pdf_path)
                metadata["page_count"] = doc.page_count

                # Extrair metadados
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float)):
                        clean_key = key.lower().replace(":", "_")
                        metadata[clean_key] = value

                # Verificar se o PDF está criptografado
                metadata["is_encrypted"] = doc.is_encrypted

                # Tamanho do arquivo
                metadata["file_size_bytes"] = os.path.getsize(pdf_path)

                doc.close()
            except Exception as e:
                logger.warning(f"Erro ao extrair metadados adicionais: {e}")

            return metadata

        except Exception as e:
            logger.error(f"Erro ao extrair metadados do PDF: {e}")
            return {"error": str(e)}

    def _extract_with_pymupdf(
        self, pdf_path: str, task_id: str, metadata: Dict[str, Any]
    ) -> List[ContentChunk]:
        """
        Extrai texto de um PDF usando PyMuPDF (fitz).

        Args:
            pdf_path: Caminho do arquivo PDF.
            task_id: ID da tarefa de coleta.
            metadata: Metadados do PDF.

        Returns:
            Lista de chunks de conteúdo.
        """
        chunks = []

        try:
            doc = fitz.open(pdf_path)

            # Extrair por página para gerar chunks
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text("text")

                # Pular páginas sem conteúdo ou com muito pouco conteúdo
                if not text or len(text.strip()) < 50:
                    continue

                # Criar chunk para a página
                chunk_id = hashlib.md5(
                    f"{pdf_path}_page{page_num}_{datetime.now().isoformat()}".encode()
                ).hexdigest()

                chunk = ContentChunk(
                    id=chunk_id,
                    task_id=task_id,
                    content_type=ContentType.TEXT,
                    text=text,
                    source_location=f"{pdf_path}#page={page_num+1}",
                    metadata={
                        "extractor": "pymupdf",
                        "page_number": page_num + 1,
                        "pdf_metadata": metadata,
                        "is_page_chunk": True,
                    },
                    length=len(text),
                    created_at=datetime.now(),
                )
                chunks.append(chunk)

            doc.close()
            return chunks

        except Exception as e:
            logger.error(f"Erro ao extrair com PyMuPDF: {e}")
            return []

    def _extract_with_pdfminer(
        self, pdf_path: str, task_id: str, metadata: Dict[str, Any]
    ) -> List[ContentChunk]:
        """
        Extrai texto de um PDF usando PDFMiner.

        Args:
            pdf_path: Caminho do arquivo PDF.
            task_id: ID da tarefa de coleta.
            metadata: Metadados do PDF.

        Returns:
            Lista de chunks de conteúdo.
        """
        chunks = []

        try:
            # PDFMiner extrai o documento inteiro
            text = pdfminer_extract_text(pdf_path)

            if not text or len(text.strip()) < 100:
                return []

            # Dividir em chunks por páginas ou seções
            if "\f" in text:  # Caractere de quebra de página
                page_texts = text.split("\f")

                for i, page_text in enumerate(page_texts):
                    if not page_text or len(page_text.strip()) < 50:
                        continue

                    chunk_id = hashlib.md5(
                        f"{pdf_path}_pdfminer_page{i}_{datetime.now().isoformat()}".encode()
                    ).hexdigest()

                    chunk = ContentChunk(
                        id=chunk_id,
                        task_id=task_id,
                        content_type=ContentType.TEXT,
                        text=page_text,
                        source_location=f"{pdf_path}#page={i+1}",
                        metadata={
                            "extractor": "pdfminer",
                            "page_number": i + 1,
                            "pdf_metadata": metadata,
                            "is_page_chunk": True,
                        },
                        length=len(page_text),
                        created_at=datetime.now(),
                    )
                    chunks.append(chunk)
            else:
                # Se não houver quebras de página, dividir em seções por títulos ou parágrafos
                # Chunks máximos de 10.000 caracteres
                MAX_CHUNK_SIZE = 10000

                # Procurar por possíveis quebras de seção (títulos, etc.)
                sections = self._split_into_sections(text, MAX_CHUNK_SIZE)

                for i, section_text in enumerate(sections):
                    if not section_text or len(section_text.strip()) < 50:
                        continue

                    chunk_id = hashlib.md5(
                        f"{pdf_path}_pdfminer_section{i}_{datetime.now().isoformat()}".encode()
                    ).hexdigest()

                    chunk = ContentChunk(
                        id=chunk_id,
                        task_id=task_id,
                        content_type=ContentType.TEXT,
                        text=section_text,
                        source_location=pdf_path,
                        metadata={
                            "extractor": "pdfminer",
                            "section_number": i + 1,
                            "pdf_metadata": metadata,
                            "is_section_chunk": True,
                        },
                        length=len(section_text),
                        created_at=datetime.now(),
                    )
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Erro ao extrair com PDFMiner: {e}")
            return []

    def _extract_with_pypdf2(
        self, pdf_path: str, task_id: str, metadata: Dict[str, Any]
    ) -> List[ContentChunk]:
        """
        Extrai texto de um PDF usando PyPDF2.

        Args:
            pdf_path: Caminho do arquivo PDF.
            task_id: ID da tarefa de coleta.
            metadata: Metadados do PDF.

        Returns:
            Lista de chunks de conteúdo.
        """
        chunks = []

        try:
            with open(pdf_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)

                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()

                    # Pular páginas sem conteúdo ou com muito pouco conteúdo
                    if not text or len(text.strip()) < 50:
                        continue

                    # Criar chunk para a página
                    chunk_id = hashlib.md5(
                        f"{pdf_path}_pypdf2_page{page_num}_{datetime.now().isoformat()}".encode()
                    ).hexdigest()

                    chunk = ContentChunk(
                        id=chunk_id,
                        task_id=task_id,
                        content_type=ContentType.TEXT,
                        text=text,
                        source_location=f"{pdf_path}#page={page_num+1}",
                        metadata={
                            "extractor": "pypdf2",
                            "page_number": page_num + 1,
                            "pdf_metadata": metadata,
                            "is_page_chunk": True,
                        },
                        length=len(text),
                        created_at=datetime.now(),
                    )
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Erro ao extrair com PyPDF2: {e}")
            return []

    def _split_into_sections(self, text: str, max_size: int = 10000) -> List[str]:
        """
        Divide um texto em seções com base em possíveis quebras de seção.

        Args:
            text: Texto para dividir.
            max_size: Tamanho máximo de cada seção.

        Returns:
            Lista de seções de texto.
        """
        # Padrão para possíveis títulos/cabeçalhos
        title_pattern = re.compile(
            r"\n\s*(?:[A-Z][A-Za-z\s]{2,50}:?|(?:\d+\.)+\s+[A-Za-z].*)\s*\n"
        )

        # Encontrar possíveis quebras de seção
        split_positions = [0]  # Começar do início

        for match in title_pattern.finditer(text):
            # Adicionar posição do título como ponto de quebra
            split_positions.append(match.start())

        split_positions.append(len(text))  # Adicionar o final do texto

        # Criar seções, garantindo que nenhuma exceda o tamanho máximo
        sections = []
        for i in range(len(split_positions) - 1):
            start = split_positions[i]
            end = split_positions[i + 1]

            # Se a seção for muito grande, dividir em partes menores
            if end - start > max_size:
                section_text = text[start:end]
                # Dividir por parágrafos
                paragraphs = section_text.split("\n\n")

                current_section = ""
                for paragraph in paragraphs:
                    if len(current_section) + len(paragraph) > max_size:
                        if current_section:
                            sections.append(current_section)
                            current_section = paragraph + "\n\n"
                        else:
                            # Se um único parágrafo for maior que max_size, dividi-lo
                            for j in range(0, len(paragraph), max_size):
                                sections.append(paragraph[j : j + max_size])
                    else:
                        current_section += paragraph + "\n\n"

                if current_section:
                    sections.append(current_section)
            else:
                sections.append(text[start:end])

        return sections
