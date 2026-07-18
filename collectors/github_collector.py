#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coletor de conteúdo do GitHub para o PipelineBuilder.

Extrai código, documentação e outros conteúdos de repositórios GitHub.
"""

import os
import re
import git
import base64
import json
import hashlib
import tempfile
import logging
import shutil
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from urllib.parse import urlparse
import requests

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
logger = APILogger("github_collector")


class GitHubCollector:
    """
    Coletor para extração de conteúdo de repositórios GitHub.
    """

    def __init__(
        self,
        storage_dir: str = "data/github_content",
        github_token: Optional[str] = None,
    ):
        """
        Inicializa o coletor GitHub.

        Args:
            storage_dir: Diretório para armazenamento temporário de conteúdo.
            github_token: Token de API do GitHub (opcional).
        """
        self.storage_dir = storage_dir
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")

        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"GitHubCollector inicializado. Storage: {storage_dir}")

    def collect(self, task: CollectionTask, source: DataSource) -> List[ContentChunk]:
        """
        Coleta conteúdo de um repositório GitHub.

        Args:
            task: Tarefa de coleta.
            source: Fonte de dados.

        Returns:
            Lista de chunks de conteúdo coletados.
        """
        logger.info(f"Iniciando coleta do repositório GitHub: {source.location}")

        if source.source_type != SourceType.GITHUB:
            logger.error(f"Tipo de fonte inválido: {source.source_type}")
            raise ValueError(
                f"GitHubCollector só pode processar fontes do tipo GITHUB, recebido: {source.source_type}"
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

            # Verificar se é uma URL de repositório ou apenas um usuário/organização
            repo_info = self._parse_github_url(source.location)

            if repo_info["type"] == "repo":
                # Coletar conteúdo de um repositório específico
                chunks = self._collect_repo(
                    repo_info["owner"],
                    repo_info["repo"],
                    task.id,
                    output_path,
                    branch=repo_info.get("branch"),
                    path=repo_info.get("path"),
                )
            elif repo_info["type"] == "user":
                # Coletar README e outros arquivos básicos de um usuário/organização
                chunks = self._collect_user_profile(
                    repo_info["owner"], task.id, output_path
                )

            # Atualizar metadados da tarefa
            task.metadata.update(
                {
                    "chunks_collected": len(chunks),
                    "total_content_length": sum(chunk.length for chunk in chunks),
                    "repo_info": repo_info,
                }
            )

            # Atualizar status da tarefa
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.content_path = output_path

            logger.info(
                f"Coleta concluída. {len(chunks)} chunks coletados do GitHub: {source.location}"
            )
            return chunks

        except Exception as e:
            logger.error(f"Erro durante coleta do GitHub {source.location}: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            raise

    def _parse_github_url(self, url: str) -> Dict[str, str]:
        """
        Analisa a URL do GitHub para extrair informações sobre o repositório.

        Args:
            url: URL do GitHub.

        Returns:
            Dicionário com informações do repositório.
        """
        # Normalizar a URL
        url = url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]

        # Extrair componentes da URL
        parsed_url = urlparse(url)

        # Se não for github.com ou github.io, lançar exceção
        if (
            "github.com" not in parsed_url.netloc
            and "github.io" not in parsed_url.netloc
        ):
            raise ValueError(f"URL não pertence ao GitHub: {url}")

        # Dividir o caminho
        path_parts = parsed_url.path.strip("/").split("/")

        result = {}

        if len(path_parts) >= 2:
            # É um repositório
            result["type"] = "repo"
            result["owner"] = path_parts[0]
            result["repo"] = path_parts[1]

            # Verificar se há uma branch ou path específico
            if len(path_parts) > 3 and path_parts[2] in ["tree", "blob"]:
                result["branch"] = path_parts[3]
                if len(path_parts) > 4:
                    result["path"] = "/".join(path_parts[4:])

        elif len(path_parts) == 1:
            # É um usuário/organização
            result["type"] = "user"
            result["owner"] = path_parts[0]
        else:
            raise ValueError(f"URL do GitHub inválida: {url}")

        return result

    def _collect_repo(
        self,
        owner: str,
        repo: str,
        task_id: str,
        output_path: str,
        branch: Optional[str] = None,
        path: Optional[str] = None,
        max_files: int = 100,
    ) -> List[ContentChunk]:
        """
        Coleta conteúdo de um repositório específico.

        Args:
            owner: Proprietário do repositório.
            repo: Nome do repositório.
            task_id: ID da tarefa de coleta.
            output_path: Caminho para salvar o conteúdo.
            branch: Branch específica para coletar.
            path: Caminho específico dentro do repositório.
            max_files: Número máximo de arquivos a serem coletados.

        Returns:
            Lista de chunks de conteúdo.
        """
        chunks = []

        # Criar diretório temporário para o clone
        temp_dir = tempfile.mkdtemp(prefix="github_clone_")

        try:
            # Obter informações do repositório via API
            repo_info = self._get_repo_info(owner, repo)

            # Determinar branch default se não especificada
            if not branch:
                branch = repo_info.get("default_branch", "main")

            # Clonar o repositório
            repo_url = f"https://github.com/{owner}/{repo}.git"
            logger.info(f"Clonando repositório {repo_url}, branch {branch}")

            try:
                # Tentar clonar apenas a branch específica para economia de banda e tempo
                git.Repo.clone_from(repo_url, temp_dir, branch=branch, depth=1)
            except git.exc.GitCommandError:
                # Se falhar, tentar novamente com a branch padrão
                logger.warning(
                    f"Falha ao clonar branch {branch}, tentando a branch padrão"
                )
                git.Repo.clone_from(repo_url, temp_dir, depth=1)

            # Adicionar README como primeiro chunk se existir
            readme_paths = [
                os.path.join(temp_dir, "README.md"),
                os.path.join(temp_dir, "README.markdown"),
                os.path.join(temp_dir, "README.rst"),
                os.path.join(temp_dir, "README"),
            ]

            for readme_path in readme_paths:
                if os.path.exists(readme_path):
                    with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                        readme_content = f.read()

                    if readme_content:
                        chunk_id = hashlib.md5(
                            f"{owner}/{repo}/README_{datetime.now().isoformat()}".encode()
                        ).hexdigest()

                        chunk = ContentChunk(
                            id=chunk_id,
                            task_id=task_id,
                            content_type=ContentType.TEXT,
                            text=readme_content,
                            source_location=f"https://github.com/{owner}/{repo}/blob/{branch}/README.md",
                            metadata={
                                "repo_owner": owner,
                                "repo_name": repo,
                                "file_path": "README.md",
                                "branch": branch,
                                "repo_info": repo_info,
                                "content_type": "documentation",
                            },
                            length=len(readme_content),
                            created_at=datetime.now(),
                        )
                        chunks.append(chunk)
                        break

            # Procurar todos os arquivos no diretório de trabalho ou subdiretório específico
            target_dir = temp_dir
            if path:
                target_path = os.path.join(temp_dir, path)
                if os.path.exists(target_path):
                    target_dir = target_path

            # Coletar todos os arquivos, filtrando por extensões de interesse
            collected_files = []
            for root, _, files in os.walk(target_dir):
                for file in files:
                    if len(collected_files) >= max_files:
                        break

                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, temp_dir)

                    # Verificar extensão e ignorar arquivos binários/ocultos
                    if self._should_process_file(rel_path):
                        collected_files.append((full_path, rel_path))

            logger.info(f"Coletados {len(collected_files)} arquivos do repositório")

            # Processar cada arquivo
            for full_path, rel_path in collected_files:
                try:
                    # Detectar tipo de conteúdo com base na extensão
                    content_type = self._detect_content_type(rel_path)

                    # Ler o arquivo
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Criar chunk
                    chunk_id = hashlib.md5(
                        f"{owner}/{repo}/{rel_path}_{datetime.now().isoformat()}".encode()
                    ).hexdigest()

                    chunk = ContentChunk(
                        id=chunk_id,
                        task_id=task_id,
                        content_type=content_type,
                        text=content,
                        source_location=f"https://github.com/{owner}/{repo}/blob/{branch}/{rel_path}",
                        metadata={
                            "repo_owner": owner,
                            "repo_name": repo,
                            "file_path": rel_path,
                            "branch": branch,
                            "file_extension": os.path.splitext(rel_path)[1],
                            "repo_info": repo_info,
                            "content_type": (
                                "code"
                                if content_type == ContentType.CODE
                                else "documentation"
                            ),
                        },
                        length=len(content),
                        created_at=datetime.now(),
                    )
                    chunks.append(chunk)

                    # Salvar o chunk
                    save_content_chunk(
                        chunk, os.path.join(output_path, f"chunk_{len(chunks)}.json")
                    )

                except Exception as e:
                    logger.warning(f"Erro ao processar arquivo {rel_path}: {e}")

            return chunks

        finally:
            # Limpar o diretório temporário
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _collect_user_profile(
        self, username: str, task_id: str, output_path: str
    ) -> List[ContentChunk]:
        """
        Coleta informações de um perfil de usuário no GitHub.

        Args:
            username: Nome do usuário.
            task_id: ID da tarefa de coleta.
            output_path: Caminho para salvar o conteúdo.

        Returns:
            Lista de chunks de conteúdo.
        """
        chunks = []

        try:
            # Obter informações do usuário via API
            user_info = self._get_user_info(username)

            # Criar chunk com o perfil do usuário
            user_text = f"# Perfil de {username} no GitHub\n\n"
            user_text += f"Nome: {user_info.get('name', 'N/A')}\n"
            user_text += f"Bio: {user_info.get('bio', 'N/A')}\n"
            user_text += f"Localização: {user_info.get('location', 'N/A')}\n"
            user_text += f"Empresa: {user_info.get('company', 'N/A')}\n"
            user_text += f"Blog: {user_info.get('blog', 'N/A')}\n"
            user_text += f"Repositórios públicos: {user_info.get('public_repos', 0)}\n"
            user_text += f"Seguidores: {user_info.get('followers', 0)}\n"
            user_text += f"Seguindo: {user_info.get('following', 0)}\n"
            user_text += f"Criado em: {user_info.get('created_at', 'N/A')}\n"

            chunk_id = hashlib.md5(
                f"github_user_{username}_{datetime.now().isoformat()}".encode()
            ).hexdigest()

            user_chunk = ContentChunk(
                id=chunk_id,
                task_id=task_id,
                content_type=ContentType.TEXT,
                text=user_text,
                source_location=f"https://github.com/{username}",
                metadata={
                    "github_username": username,
                    "user_info": user_info,
                    "content_type": "profile",
                },
                length=len(user_text),
                created_at=datetime.now(),
            )
            chunks.append(user_chunk)

            # Obter lista de repositórios do usuário
            repos = self._get_user_repos(username)

            if repos:
                # Criar chunk com lista de repositórios
                repos_text = f"# Repositórios de {username}\n\n"

                for i, repo in enumerate(repos[:30]):  # Limitar a 30 repositórios
                    repos_text += f"## {i+1}. {repo.get('name')}\n\n"
                    repos_text += f"- Descrição: {repo.get('description', 'N/A')}\n"
                    repos_text += (
                        f"- Linguagem principal: {repo.get('language', 'N/A')}\n"
                    )
                    repos_text += f"- Estrelas: {repo.get('stargazers_count', 0)}\n"
                    repos_text += f"- Forks: {repo.get('forks_count', 0)}\n"
                    repos_text += f"- Criado em: {repo.get('created_at', 'N/A')}\n"
                    repos_text += (
                        f"- Última atualização: {repo.get('updated_at', 'N/A')}\n"
                    )
                    repos_text += f"- URL: {repo.get('html_url', 'N/A')}\n\n"

                chunk_id = hashlib.md5(
                    f"github_user_repos_{username}_{datetime.now().isoformat()}".encode()
                ).hexdigest()

                repos_chunk = ContentChunk(
                    id=chunk_id,
                    task_id=task_id,
                    content_type=ContentType.TEXT,
                    text=repos_text,
                    source_location=f"https://github.com/{username}?tab=repositories",
                    metadata={
                        "github_username": username,
                        "content_type": "repositories_list",
                        "repos_count": len(repos),
                    },
                    length=len(repos_text),
                    created_at=datetime.now(),
                )
                chunks.append(repos_chunk)

            # Salvar os chunks
            for i, chunk in enumerate(chunks):
                save_content_chunk(chunk, os.path.join(output_path, f"chunk_{i}.json"))

            return chunks

        except Exception as e:
            logger.error(f"Erro ao coletar perfil do usuário {username}: {e}")
            raise

    def _get_repo_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Obtém informações de um repositório via API do GitHub.

        Args:
            owner: Proprietário do repositório.
            repo: Nome do repositório.

        Returns:
            Dicionário com informações do repositório.
        """
        url = f"https://api.github.com/repos/{owner}/{repo}"

        headers = {"Accept": "application/vnd.github.v3+json"}

        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(
                f"Falha ao obter informações do repositório {owner}/{repo}: {response.status_code}"
            )
            return {"error": f"Status code: {response.status_code}"}

    def _get_user_info(self, username: str) -> Dict[str, Any]:
        """
        Obtém informações de um usuário via API do GitHub.

        Args:
            username: Nome do usuário.

        Returns:
            Dicionário com informações do usuário.
        """
        url = f"https://api.github.com/users/{username}"

        headers = {"Accept": "application/vnd.github.v3+json"}

        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(
                f"Falha ao obter informações do usuário {username}: {response.status_code}"
            )
            return {"error": f"Status code: {response.status_code}"}

    def _get_user_repos(self, username: str) -> List[Dict[str, Any]]:
        """
        Obtém repositórios de um usuário via API do GitHub.

        Args:
            username: Nome do usuário.

        Returns:
            Lista de dicionários com informações dos repositórios.
        """
        url = f"https://api.github.com/users/{username}/repos"

        headers = {"Accept": "application/vnd.github.v3+json"}

        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        params = {"sort": "updated", "per_page": 100}

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(
                f"Falha ao obter repositórios do usuário {username}: {response.status_code}"
            )
            return []

    def _should_process_file(self, file_path: str) -> bool:
        """
        Verifica se um arquivo deve ser processado com base na extensão e caminho.

        Args:
            file_path: Caminho do arquivo.

        Returns:
            True se o arquivo deve ser processado, False caso contrário.
        """
        # Ignorar diretórios ocultos e arquivos ocultos
        if any(part.startswith(".") for part in file_path.split("/")):
            return False

        # Ignorar arquivos binários comuns
        binary_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".ico",
            ".svg",
            ".mp3",
            ".mp4",
            ".avi",
            ".mov",
            ".pdf",
            ".zip",
            ".tar.gz",
            ".jar",
            ".war",
            ".ear",
            ".class",
            ".pyc",
            ".so",
            ".dll",
            ".exe",
        }

        _, ext = os.path.splitext(file_path.lower())
        if ext in binary_extensions:
            return False

        # Ignorar diretórios node_modules, venv, etc.
        ignored_dirs = {
            "node_modules/",
            "venv/",
            "env/",
            "__pycache__/",
            "build/",
            "dist/",
        }
        if any(ignored_dir in file_path for ignored_dir in ignored_dirs):
            return False

        # Verificar tamanho do arquivo (em uma implementação real)
        # Para este exemplo, consideramos que todos os arquivos têm tamanho aceitável

        return True

    def _detect_content_type(self, file_path: str) -> ContentType:
        """
        Detecta o tipo de conteúdo com base na extensão do arquivo.

        Args:
            file_path: Caminho do arquivo.

        Returns:
            Tipo de conteúdo.
        """
        _, ext = os.path.splitext(file_path.lower())

        # Extensões de código-fonte
        code_extensions = {
            ".py",
            ".java",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".c",
            ".cpp",
            ".h",
            ".cs",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".html",
            ".css",
            ".scss",
            ".json",
            ".xml",
            ".yaml",
            ".yml",
            ".sh",
            ".bash",
            ".sql",
            ".kt",
            ".swift",
            ".m",
            ".scala",
            ".clj",
            ".ex",
            ".exs",
            ".erl",
            ".fs",
            ".f90",
            ".r",
            ".dart",
        }

        # Extensões de documentação
        doc_extensions = {
            ".md",
            ".markdown",
            ".rst",
            ".txt",
            ".org",
            ".asciidoc",
            ".adoc",
            ".tex",
            ".wiki",
            ".rdoc",
            ".pod",
            ".man",
        }

        if ext in code_extensions:
            return ContentType.CODE
        elif ext in doc_extensions:
            return ContentType.TEXT
        else:
            return ContentType.TEXT  # Padrão para tipos desconhecidos
