#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testes abrangentes para os coletores do PipelineBuilder.

Este módulo contém testes para verificar a funcionalidade
dos diferentes coletores implementados.
"""

import os
import sys
import uuid
import json
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Adicionar diretório pai ao path para importação
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from cerberus_api.pipeline_builder.models import (
    DataSource,
    CollectionTask,
    SourceType,
    ContentType,
    TaskStatus,
)
from cerberus_api.pipeline_builder.collectors import (
    WebCollector,
    PDFCollector,
    GitHubCollector,
    YouTubeCollector,
)


class TestBaseCollector:
    """
    Classe base para testes de coletores.
    """

    def setUp(self):
        """
        Configuração para os testes.
        """
        # Criar diretório temporário para testes
        self.test_dir = Path("test_output/collectors")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Criar tarefa de coleta genérica
        self.task = CollectionTask(
            id=str(uuid.uuid4()),
            source_id=str(uuid.uuid4()),
            status=TaskStatus.PENDING,
            metadata={},
            created_at=datetime.now(),
        )

    def tearDown(self):
        """
        Limpeza após os testes.
        """
        # Opcionalmente remover diretório de testes
        # import shutil
        # shutil.rmtree(self.test_dir, ignore_errors=True)
        pass

    def assert_chunks_valid(self, chunks):
        """
        Verifica se os chunks coletados são válidos.
        """
        self.assertIsNotNone(chunks)
        self.assertIsInstance(chunks, list)

        if chunks:
            for chunk in chunks:
                self.assertIsNotNone(chunk.id)
                self.assertEqual(chunk.task_id, self.task.id)
                self.assertIsNotNone(chunk.text)
                self.assertIsNotNone(chunk.source_location)
                self.assertGreater(chunk.length, 0)
                self.assertIsNotNone(chunk.created_at)


class TestWebCollector(unittest.TestCase, TestBaseCollector):
    """
    Testes para o WebCollector.
    """

    def setUp(self):
        """
        Configuração para os testes do WebCollector.
        """
        TestBaseCollector.setUp(self)
        self.collector = WebCollector(storage_dir=str(self.test_dir / "web_content"))

    @patch("cerberus_api.pipeline_builder.collectors.web_collector.requests.get")
    def test_collect_simple_page(self, mock_get):
        """
        Testa a coleta de uma página web simples.
        """
        # Configurar mock para simular resposta HTTP
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><h1>Título de Teste</h1><p>Conteúdo de teste para o WebCollector.</p></body></html>"
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_get.return_value = mock_response

        # Criar fonte de dados
        source = DataSource(
            id=str(uuid.uuid4()),
            name="Página de Teste",
            source_type=SourceType.WEB,
            location="https://example.com/test",
            description="Página de teste para o WebCollector",
            metadata={},
            created_at=datetime.now(),
        )

        # Executar coleta
        chunks = self.collector.collect(self.task, source)

        # Verificar resultados
        self.assert_chunks_valid(chunks)
        self.assertEqual(self.task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(self.task.completed_at)

        # Verificar conteúdo específico
        if chunks:
            self.assertIn("Título de Teste", chunks[0].text)
            self.assertIn("Conteúdo de teste", chunks[0].text)


class TestPDFCollector(unittest.TestCase, TestBaseCollector):
    """
    Testes para o PDFCollector.
    """

    def setUp(self):
        """
        Configuração para os testes do PDFCollector.
        """
        TestBaseCollector.setUp(self)
        self.collector = PDFCollector(storage_dir=str(self.test_dir / "pdf_content"))

    @patch("cerberus_api.pipeline_builder.collectors.pdf_collector.requests.get")
    @patch("cerberus_api.pipeline_builder.collectors.pdf_collector.PyPDF2.PdfReader")
    @patch("cerberus_api.pipeline_builder.collectors.pdf_collector.fitz.open")
    @patch(
        "cerberus_api.pipeline_builder.collectors.pdf_collector.pdfminer_extract_text"
    )
    def test_collect_pdf_from_url(
        self, mock_pdfminer, mock_fitz_open, mock_pdf_reader, mock_get
    ):
        """
        Testa a coleta de um PDF a partir de uma URL.
        """
        # Configurar mocks
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.iter_content.return_value = [b"PDF content"]
        mock_get.return_value = mock_response

        # Mock para PyPDF2
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Texto extraído do PDF com PyPDF2."
        mock_pdf_reader.return_value.pages = [mock_page]
        mock_pdf_reader.return_value.metadata = {"Title": "Documento de Teste"}

        # Mock para PyMuPDF (fitz)
        mock_doc = MagicMock()
        mock_page_fitz = MagicMock()
        mock_page_fitz.get_text.return_value = "Texto extraído do PDF com PyMuPDF."
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = mock_page_fitz
        mock_doc.metadata = {"title": "Documento de Teste"}
        mock_fitz_open.return_value = mock_doc

        # Mock para PDFMiner
        mock_pdfminer.return_value = "Texto extraído do PDF com PDFMiner."

        # Criar fonte de dados
        source = DataSource(
            id=str(uuid.uuid4()),
            name="PDF de Teste",
            source_type=SourceType.PDF,
            location="https://example.com/test.pdf",
            description="PDF de teste para o PDFCollector",
            metadata={},
            created_at=datetime.now(),
        )

        # Executar coleta
        with patch("builtins.open", unittest.mock.mock_open()):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.getsize", return_value=1024):
                    chunks = self.collector.collect(self.task, source)

        # Verificar resultados
        self.assert_chunks_valid(chunks)
        self.assertEqual(self.task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(self.task.completed_at)


class TestGitHubCollector(unittest.TestCase, TestBaseCollector):
    """
    Testes para o GitHubCollector.
    """

    def setUp(self):
        """
        Configuração para os testes do GitHubCollector.
        """
        TestBaseCollector.setUp(self)
        self.collector = GitHubCollector(
            storage_dir=str(self.test_dir / "github_content")
        )

    @patch("cerberus_api.pipeline_builder.collectors.github_collector.requests.get")
    @patch(
        "cerberus_api.pipeline_builder.collectors.github_collector.git.Repo.clone_from"
    )
    def test_collect_github_repository(self, mock_clone, mock_get):
        """
        Testa a coleta de um repositório GitHub.
        """
        # Configurar mocks
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "test-repo",
            "description": "Repositório de teste",
            "default_branch": "main",
            "owner": {"login": "test-user"},
            "html_url": "https://github.com/test-user/test-repo",
        }
        mock_get.return_value = mock_response

        # Mock para clone de repositório
        mock_clone.return_value = MagicMock()

        # Criar fonte de dados
        source = DataSource(
            id=str(uuid.uuid4()),
            name="Repositório de Teste",
            source_type=SourceType.GITHUB,
            location="https://github.com/test-user/test-repo",
            description="Repositório de teste para o GitHubCollector",
            metadata={},
            created_at=datetime.now(),
        )

        # Executar coleta com mocks para sistema de arquivos
        with patch("os.walk") as mock_walk:
            mock_walk.return_value = [
                (str(self.test_dir), [], ["README.md"]),
                (str(self.test_dir / "src"), [], ["example.py"]),
            ]

            with patch("os.path.exists", return_value=True):
                with patch("os.path.relpath", return_value="README.md"):
                    with patch(
                        "builtins.open",
                        unittest.mock.mock_open(
                            read_data="# Projeto de Teste\nEste é um repositório de teste."
                        ),
                    ):
                        with patch("os.path.isfile", return_value=True):
                            chunks = self.collector.collect(self.task, source)

        # Verificar resultados
        self.assertEqual(self.task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(self.task.completed_at)


class TestYouTubeCollector(unittest.TestCase, TestBaseCollector):
    """
    Testes para o YouTubeCollector.
    """

    def setUp(self):
        """
        Configuração para os testes do YouTubeCollector.
        """
        TestBaseCollector.setUp(self)
        self.collector = YouTubeCollector(
            storage_dir=str(self.test_dir / "youtube_content")
        )

    def test_collect_youtube_video_no_api_key(self):
        """
        Testa a coleta de um vídeo do YouTube sem chave de API.
        """
        # Criar fonte de dados
        source = DataSource(
            id=str(uuid.uuid4()),
            name="Vídeo de Teste",
            source_type=SourceType.YOUTUBE,
            location="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            description="Vídeo de teste para o YouTubeCollector",
            metadata={},
            created_at=datetime.now(),
        )

        # Executar coleta
        chunks = self.collector.collect(self.task, source)

        # Verificar resultados
        self.assert_chunks_valid(chunks)
        self.assertEqual(self.task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(self.task.completed_at)

        # Verificar que pelo menos um chunk foi gerado (metadados simulados)
        self.assertGreaterEqual(len(chunks), 1)
        if chunks:
            # Deve conter um aviso sobre API não disponível
            self.assertIn("API", chunks[0].text)

    @patch("cerberus_api.pipeline_builder.collectors.youtube_collector.requests.get")
    def test_collect_youtube_video_with_api_key(self, mock_get):
        """
        Testa a coleta de um vídeo do YouTube com chave de API.
        """
        # Configurar mocks
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "snippet": {
                        "title": "Vídeo de Teste",
                        "description": "Descrição do vídeo de teste",
                        "publishedAt": "2023-01-01T00:00:00Z",
                        "channelTitle": "Canal de Teste",
                        "channelId": "UC123456789",
                        "tags": ["teste", "youtube"],
                    },
                    "contentDetails": {"duration": "PT5M30S"},
                    "statistics": {
                        "viewCount": "1000",
                        "likeCount": "100",
                        "commentCount": "50",
                    },
                }
            ]
        }
        mock_get.return_value = mock_response

        # Criar fonte de dados
        source = DataSource(
            id=str(uuid.uuid4()),
            name="Vídeo de Teste",
            source_type=SourceType.YOUTUBE,
            location="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            description="Vídeo de teste para o YouTubeCollector",
            metadata={},
            created_at=datetime.now(),
        )

        # Configurar coletor com chave de API simulada
        self.collector.api_key = "fake_api_key"

        # Executar coleta
        chunks = self.collector.collect(self.task, source)

        # Verificar resultados
        self.assert_chunks_valid(chunks)
        self.assertEqual(self.task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(self.task.completed_at)

        # Verificar que pelo menos dois chunks foram gerados (metadados + transcrição)
        self.assertGreaterEqual(len(chunks), 2)
        if len(chunks) >= 2:
            # Verificar conteúdo específico
            metadata_chunk = chunks[0]
            transcript_chunk = chunks[1]

            self.assertIn("Vídeo de Teste", metadata_chunk.text)
            self.assertIn("transcrição", transcript_chunk.text.lower())


class TestMultiSource(unittest.TestCase):
    """
    Testes integrados para múltiplas fontes.
    """

    def setUp(self):
        """
        Configuração para os testes integrados.
        """
        # Criar diretório temporário para testes
        self.test_dir = Path("test_output/multi_source")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar coletores
        self.web_collector = WebCollector(
            storage_dir=str(self.test_dir / "web_content")
        )
        self.pdf_collector = PDFCollector(
            storage_dir=str(self.test_dir / "pdf_content")
        )
        self.github_collector = GitHubCollector(
            storage_dir=str(self.test_dir / "github_content")
        )
        self.youtube_collector = YouTubeCollector(
            storage_dir=str(self.test_dir / "youtube_content")
        )

    @patch("cerberus_api.pipeline_builder.collectors.web_collector.requests.get")
    @patch("cerberus_api.pipeline_builder.collectors.pdf_collector.requests.get")
    @patch("cerberus_api.pipeline_builder.collectors.youtube_collector.requests.get")
    def test_collect_from_multiple_sources(
        self, mock_youtube_get, mock_pdf_get, mock_web_get
    ):
        """
        Testa a coleta de diferentes tipos de fontes.
        """
        # Configurar mocks para cada tipo de coletor
        # Web
        mock_web_response = MagicMock()
        mock_web_response.status_code = 200
        mock_web_response.text = "<html><body><h1>Página de Teste</h1><p>Conteúdo da página web.</p></body></html>"
        mock_web_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_web_get.return_value = mock_web_response

        # PDF
        mock_pdf_response = MagicMock()
        mock_pdf_response.status_code = 200
        mock_pdf_response.headers = {"Content-Type": "application/pdf"}
        mock_pdf_response.iter_content.return_value = [b"PDF content"]
        mock_pdf_get.return_value = mock_pdf_response

        # YouTube
        mock_youtube_response = MagicMock()
        mock_youtube_response.status_code = 200
        mock_youtube_response.json.return_value = {
            "items": [
                {
                    "snippet": {
                        "title": "Vídeo de Teste",
                        "description": "Descrição do vídeo",
                        "publishedAt": "2023-01-01T00:00:00Z",
                        "channelTitle": "Canal de Teste",
                    },
                    "contentDetails": {"duration": "PT5M30S"},
                    "statistics": {"viewCount": "1000", "likeCount": "100"},
                }
            ]
        }
        mock_youtube_get.return_value = mock_youtube_response

        # Criar tarefas e fontes
        results = {}

        # Testar coleta de web
        web_source = DataSource(
            id=str(uuid.uuid4()),
            name="Página Web",
            source_type=SourceType.WEB,
            location="https://example.com/test",
            description="Teste de página web",
            metadata={},
            created_at=datetime.now(),
        )

        web_task = CollectionTask(
            id=str(uuid.uuid4()),
            source_id=web_source.id,
            status=TaskStatus.PENDING,
            metadata={},
            created_at=datetime.now(),
        )

        # Testar coleta de PDF
        pdf_source = DataSource(
            id=str(uuid.uuid4()),
            name="Documento PDF",
            source_type=SourceType.PDF,
            location="https://example.com/test.pdf",
            description="Teste de documento PDF",
            metadata={},
            created_at=datetime.now(),
        )

        pdf_task = CollectionTask(
            id=str(uuid.uuid4()),
            source_id=pdf_source.id,
            status=TaskStatus.PENDING,
            metadata={},
            created_at=datetime.now(),
        )

        # Testar coleta de YouTube
        youtube_source = DataSource(
            id=str(uuid.uuid4()),
            name="Vídeo YouTube",
            source_type=SourceType.YOUTUBE,
            location="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            description="Teste de vídeo do YouTube",
            metadata={},
            created_at=datetime.now(),
        )

        youtube_task = CollectionTask(
            id=str(uuid.uuid4()),
            source_id=youtube_source.id,
            status=TaskStatus.PENDING,
            metadata={},
            created_at=datetime.now(),
        )

        # Executar coletas com mocks apropriados
        with patch("builtins.open", unittest.mock.mock_open()):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.getsize", return_value=1024):
                    # Coletar da web
                    web_chunks = self.web_collector.collect(web_task, web_source)
                    results["web"] = {"chunks": web_chunks, "task": web_task}

                    # Coletar PDF
                    with patch(
                        "cerberus_api.pipeline_builder.collectors.pdf_collector.PyPDF2.PdfReader"
                    ):
                        with patch(
                            "cerberus_api.pipeline_builder.collectors.pdf_collector.fitz.open"
                        ):
                            with patch(
                                "cerberus_api.pipeline_builder.collectors.pdf_collector.pdfminer_extract_text"
                            ):
                                pdf_chunks = self.pdf_collector.collect(
                                    pdf_task, pdf_source
                                )
                                results["pdf"] = {
                                    "chunks": pdf_chunks,
                                    "task": pdf_task,
                                }

                    # Coletar YouTube
                    self.youtube_collector.api_key = "fake_api_key"
                    youtube_chunks = self.youtube_collector.collect(
                        youtube_task, youtube_source
                    )
                    results["youtube"] = {
                        "chunks": youtube_chunks,
                        "task": youtube_task,
                    }

        # Verificar resultados de todas as coletas
        for source_type, result in results.items():
            chunks = result["chunks"]
            task = result["task"]

            # Verificar status da tarefa
            self.assertEqual(task.status, TaskStatus.COMPLETED)
            self.assertIsNotNone(task.completed_at)

            # Verificar chunks
            self.assertIsNotNone(chunks)
            self.assertIsInstance(chunks, list)
            self.assertGreater(len(chunks), 0)

            # Verificar que os chunks têm os campos necessários
            for chunk in chunks:
                self.assertIsNotNone(chunk.id)
                self.assertEqual(chunk.task_id, task.id)
                self.assertIsNotNone(chunk.text)
                self.assertIsNotNone(chunk.source_location)
                self.assertGreater(chunk.length, 0)
                self.assertIsNotNone(chunk.created_at)


if __name__ == "__main__":
    unittest.main()
