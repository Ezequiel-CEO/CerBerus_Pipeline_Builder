#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ferramentas de manipulação de arquivos para o assistente Èter.

Este módulo fornece funcionalidades para ler, escrever, editar e gerenciar
arquivos e diretórios no sistema.
"""

import os
import sys
import json
import shutil
import tempfile
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from cerberus_api.utils.logging_config import get_logger

logger = get_logger("eter_files")


class FileCommands:
    """
    Classe para operações com arquivos e diretórios.
    """

    @staticmethod
    def read_file(
        file_path: str,
        encoding: str = "utf-8",
        max_size: int = 10 * 1024 * 1024,  # 10MB
    ) -> Dict[str, Any]:
        """
        Lê o conteúdo de um arquivo.

        Args:
            file_path: Caminho do arquivo
            encoding: Codificação para leitura do arquivo
            max_size: Tamanho máximo para leitura (em bytes)

        Returns:
            Dicionário com o conteúdo do arquivo
        """
        logger.info(f"Lendo arquivo: {file_path}")

        try:
            path = Path(file_path)

            # Verificar existência
            if not path.exists():
                return {
                    "success": False,
                    "error": "file_not_found",
                    "message": f"Arquivo não encontrado: {file_path}",
                }

            # Verificar se é arquivo
            if not path.is_file():
                return {
                    "success": False,
                    "error": "not_a_file",
                    "message": f"O caminho especificado não é um arquivo: {file_path}",
                }

            # Verificar tamanho
            file_size = path.stat().st_size
            if file_size > max_size:
                logger.warning(f"Arquivo muito grande ({file_size} bytes): {file_path}")
                return {
                    "success": False,
                    "error": "file_too_large",
                    "message": f"Arquivo excede o tamanho máximo permitido ({max_size} bytes)",
                    "file_size": file_size,
                    "max_size": max_size,
                }

            # Ler o arquivo
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            return {
                "success": True,
                "content": content,
                "file_path": str(path.absolute()),
                "size": file_size,
                "encoding": encoding,
            }

        except UnicodeDecodeError:
            logger.warning(f"Erro de decodificação para {encoding}: {file_path}")
            return {
                "success": False,
                "error": "encoding_error",
                "message": f"Não foi possível decodificar o arquivo usando a codificação {encoding}",
            }

        except Exception as e:
            logger.error(f"Erro ao ler arquivo '{file_path}': {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao ler arquivo: {str(e)}",
            }

    @staticmethod
    def write_file(
        file_path: str,
        content: str,
        append: bool = False,
        encoding: str = "utf-8",
        make_dirs: bool = True,
        backup: bool = False,
    ) -> Dict[str, Any]:
        """
        Escreve conteúdo em um arquivo.

        Args:
            file_path: Caminho do arquivo
            content: Conteúdo a ser escrito
            append: Se deve adicionar ao final do arquivo
            encoding: Codificação para escrita
            make_dirs: Cria diretórios intermediários se necessário
            backup: Cria backup do arquivo original se já existir

        Returns:
            Dicionário com o resultado da operação
        """
        logger.info(f"Escrevendo arquivo: {file_path} (append={append})")

        try:
            path = Path(file_path)

            # Criar diretórios se necessário
            if make_dirs:
                os.makedirs(path.parent, exist_ok=True)

            # Fazer backup se solicitado e arquivo existir
            backup_path = None
            if backup and path.exists() and path.is_file():
                backup_path = f"{file_path}.bak"
                shutil.copy2(file_path, backup_path)
                logger.debug(f"Backup criado: {backup_path}")

            # Modo de abertura
            mode = "a" if append else "w"

            # Escrever o arquivo
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)

            # Obter informações do arquivo escrito
            file_size = path.stat().st_size

            result = {
                "success": True,
                "file_path": str(path.absolute()),
                "size": file_size,
                "append": append,
                "encoding": encoding,
            }

            # Adicionar informações do backup
            if backup_path:
                result["backup"] = backup_path

            return result

        except Exception as e:
            logger.error(f"Erro ao escrever arquivo '{file_path}': {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao escrever arquivo: {str(e)}",
            }

    @staticmethod
    def delete_file(
        file_path: str, force: bool = False, recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Remove um arquivo ou diretório.

        Args:
            file_path: Caminho do arquivo/diretório
            force: Se deve ignorar erros
            recursive: Se deve remover diretórios recursivamente

        Returns:
            Dicionário com o resultado da operação
        """
        logger.info(f"Removendo: {file_path} (force={force}, recursive={recursive})")

        try:
            path = Path(file_path)

            # Verificar existência
            if not path.exists():
                return {
                    "success": False,
                    "error": "not_found",
                    "message": f"Arquivo/diretório não encontrado: {file_path}",
                }

            # Armazenar informações antes de remover
            is_dir = path.is_dir()
            file_info = {
                "path": str(path.absolute()),
                "is_directory": is_dir,
                "size": path.stat().st_size if not is_dir else None,
            }

            # Remover diretório ou arquivo
            if is_dir:
                if recursive:
                    shutil.rmtree(path, ignore_errors=force)
                else:
                    path.rmdir()  # Falha se não estiver vazio
            else:
                path.unlink(missing_ok=force)

            return {
                "success": True,
                "file_info": file_info,
                "message": f"{'Diretório' if is_dir else 'Arquivo'} removido: {file_path}",
            }

        except Exception as e:
            logger.error(f"Erro ao remover '{file_path}': {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao remover: {str(e)}",
            }

    @staticmethod
    def list_directory(
        dir_path: str,
        pattern: Optional[str] = None,
        recursive: bool = False,
        show_hidden: bool = False,
    ) -> Dict[str, Any]:
        """
        Lista o conteúdo de um diretório.

        Args:
            dir_path: Caminho do diretório
            pattern: Padrão glob para filtrar arquivos
            recursive: Se deve listar recursivamente
            show_hidden: Se deve incluir arquivos/diretórios ocultos

        Returns:
            Dicionário com a listagem do diretório
        """
        logger.info(
            f"Listando diretório: {dir_path} (pattern={pattern}, recursive={recursive})"
        )

        try:
            path = Path(dir_path)

            # Verificar existência
            if not path.exists():
                return {
                    "success": False,
                    "error": "not_found",
                    "message": f"Diretório não encontrado: {dir_path}",
                }

            # Verificar se é diretório
            if not path.is_dir():
                return {
                    "success": False,
                    "error": "not_a_directory",
                    "message": f"O caminho não é um diretório: {dir_path}",
                }

            # Listar conteúdo
            if recursive:
                # Usar glob para listagem recursiva com padrão
                if pattern:
                    glob_pattern = os.path.join(dir_path, "**", pattern)
                    items = glob.glob(glob_pattern, recursive=True)
                else:
                    items = []
                    for root, dirs, files in os.walk(dir_path):
                        for name in files + dirs:
                            items.append(os.path.join(root, name))
            else:
                # Listar apenas o diretório atual
                items = path.iterdir()
                if pattern:
                    items = path.glob(pattern)
                else:
                    items = list(path.iterdir())

            # Filtrar itens ocultos se necessário
            if not show_hidden:
                items = [
                    item for item in items if not os.path.basename(item).startswith(".")
                ]

            # Preparar resultado
            files = []
            directories = []

            for item in items:
                item_path = Path(item)
                item_info = {
                    "name": item_path.name,
                    "path": str(item_path.absolute()),
                    "relative_path": (
                        str(item_path.relative_to(path)) if item_path != path else "."
                    ),
                }

                if item_path.is_dir():
                    directories.append(item_info)
                else:
                    # Adicionar informações extra para arquivos
                    item_info["size"] = item_path.stat().st_size
                    item_info["extension"] = item_path.suffix
                    files.append(item_info)

            return {
                "success": True,
                "directory": str(path.absolute()),
                "files": files,
                "directories": directories,
                "count": {
                    "files": len(files),
                    "directories": len(directories),
                    "total": len(files) + len(directories),
                },
            }

        except Exception as e:
            logger.error(f"Erro ao listar diretório '{dir_path}': {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao listar diretório: {str(e)}",
            }

    @staticmethod
    def create_directory(
        dir_path: str, mode: int = 0o755, exist_ok: bool = True
    ) -> Dict[str, Any]:
        """
        Cria um diretório.

        Args:
            dir_path: Caminho do diretório
            mode: Permissões do diretório
            exist_ok: Se deve ignorar se já existir

        Returns:
            Dicionário com o resultado da operação
        """
        logger.info(f"Criando diretório: {dir_path}")

        try:
            path = Path(dir_path)

            # Verificar se já existe
            if path.exists():
                if path.is_dir():
                    if not exist_ok:
                        return {
                            "success": False,
                            "error": "directory_exists",
                            "message": f"Diretório já existe: {dir_path}",
                        }
                    else:
                        return {
                            "success": True,
                            "directory": str(path.absolute()),
                            "message": f"Diretório já existe: {dir_path}",
                            "created": False,
                        }
                else:
                    return {
                        "success": False,
                        "error": "path_exists_not_dir",
                        "message": f"O caminho existe, mas não é um diretório: {dir_path}",
                    }

            # Criar diretório
            os.makedirs(dir_path, mode=mode, exist_ok=exist_ok)

            return {
                "success": True,
                "directory": str(path.absolute()),
                "message": f"Diretório criado: {dir_path}",
                "created": True,
            }

        except Exception as e:
            logger.error(f"Erro ao criar diretório '{dir_path}': {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao criar diretório: {str(e)}",
            }

    @staticmethod
    def copy(
        source: str, destination: str, recursive: bool = True, overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Copia arquivo ou diretório.

        Args:
            source: Caminho da origem
            destination: Caminho do destino
            recursive: Se deve copiar diretórios recursivamente
            overwrite: Se deve sobrescrever arquivos existentes

        Returns:
            Dicionário com o resultado da operação
        """
        logger.info(f"Copiando: {source} -> {destination}")

        try:
            src_path = Path(source)
            dst_path = Path(destination)

            # Verificar existência da origem
            if not src_path.exists():
                return {
                    "success": False,
                    "error": "source_not_found",
                    "message": f"Arquivo/diretório de origem não encontrado: {source}",
                }

            # Verificar se destino já existe
            if dst_path.exists() and not overwrite:
                return {
                    "success": False,
                    "error": "destination_exists",
                    "message": f"Destino já existe: {destination}. Use overwrite=True para sobrescrever.",
                }

            # Copiar arquivo ou diretório
            if src_path.is_dir():
                if recursive:
                    # Copiar diretório recursivamente
                    if dst_path.exists() and overwrite:
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path)
                else:
                    return {
                        "success": False,
                        "error": "source_is_directory",
                        "message": f"Origem é um diretório. Use recursive=True para copiar recursivamente.",
                    }
            else:
                # Copiar arquivo
                # Criar diretórios de destino se necessário
                os.makedirs(dst_path.parent, exist_ok=True)
                shutil.copy2(src_path, dst_path)

            return {
                "success": True,
                "source": str(src_path.absolute()),
                "destination": str(dst_path.absolute()),
                "type": "directory" if src_path.is_dir() else "file",
                "message": f"{'Diretório' if src_path.is_dir() else 'Arquivo'} copiado com sucesso",
            }

        except Exception as e:
            logger.error(f"Erro ao copiar '{source}' para '{destination}': {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao copiar: {str(e)}",
            }

    @staticmethod
    def move(source: str, destination: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Move arquivo ou diretório.

        Args:
            source: Caminho da origem
            destination: Caminho do destino
            overwrite: Se deve sobrescrever arquivos existentes

        Returns:
            Dicionário com o resultado da operação
        """
        logger.info(f"Movendo: {source} -> {destination}")

        try:
            src_path = Path(source)
            dst_path = Path(destination)

            # Verificar existência da origem
            if not src_path.exists():
                return {
                    "success": False,
                    "error": "source_not_found",
                    "message": f"Arquivo/diretório de origem não encontrado: {source}",
                }

            # Verificar se destino já existe
            if dst_path.exists() and not overwrite:
                return {
                    "success": False,
                    "error": "destination_exists",
                    "message": f"Destino já existe: {destination}. Use overwrite=True para sobrescrever.",
                }

            # Criar diretórios de destino se necessário
            os.makedirs(dst_path.parent, exist_ok=True)

            # Remover destino se existir e overwrite=True
            if dst_path.exists() and overwrite:
                if dst_path.is_dir():
                    shutil.rmtree(dst_path)
                else:
                    dst_path.unlink()

            # Mover arquivo ou diretório
            shutil.move(str(src_path), str(dst_path))

            return {
                "success": True,
                "source": str(src_path),
                "destination": str(dst_path.absolute()),
                "type": "directory" if src_path.is_dir() else "file",
                "message": f"{'Diretório' if src_path.is_dir() else 'Arquivo'} movido com sucesso",
            }

        except Exception as e:
            logger.error(f"Erro ao mover '{source}' para '{destination}': {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao mover: {str(e)}",
            }


# Funções auxiliares para uso direto do módulo
def read_file(file_path: str, **kwargs) -> Dict[str, Any]:
    """Wrapper para leitura rápida de arquivos."""
    return FileCommands.read_file(file_path, **kwargs)


def write_file(file_path: str, content: str, **kwargs) -> Dict[str, Any]:
    """Wrapper para escrita rápida de arquivos."""
    return FileCommands.write_file(file_path, content, **kwargs)


def list_dir(dir_path: str, **kwargs) -> Dict[str, Any]:
    """Wrapper para listagem rápida de diretórios."""
    return FileCommands.list_directory(dir_path, **kwargs)
