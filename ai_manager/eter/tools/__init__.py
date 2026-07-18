#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ferramentas para o assistente Èter.

Este módulo fornece um conjunto de ferramentas específicas que o assistente
Èter pode usar para interagir com o sistema, manipular dados e executar tarefas.
"""

from typing import Dict, List, Any, Optional, Callable, Union
import os
import sys
import json
import subprocess
from pathlib import Path

__all__ = [
    "ShellTool",
    "FileTool",
]


class ShellTool:
    """Ferramenta para execução de comandos no shell."""

    @staticmethod
    def execute(
        command: str,
        cwd: Optional[str] = None,
        capture_output: bool = True,
        timeout: Optional[int] = None,
        shell: bool = True,
    ) -> Dict[str, Any]:
        """
        Executa um comando no shell.

        Args:
            command: Comando a ser executado
            cwd: Diretório de trabalho
            capture_output: Se deve capturar saída stdout/stderr
            timeout: Tempo limite em segundos
            shell: Se deve usar shell para execução

        Returns:
            Dicionário com o resultado da execução
        """
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=capture_output,
                text=True,
                cwd=cwd,
                timeout=timeout,
            )

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout if capture_output else None,
                "stderr": result.stderr if capture_output else None,
                "command": command,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "timeout",
                "message": f"Comando excedeu o tempo limite de {timeout}s",
                "command": command,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao executar comando: {str(e)}",
                "command": command,
            }


class FileTool:
    """Ferramenta para operações com arquivos."""

    @staticmethod
    def read(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """
        Lê o conteúdo de um arquivo.

        Args:
            file_path: Caminho do arquivo
            encoding: Codificação do arquivo

        Returns:
            Dicionário com o resultado da operação
        """
        try:
            path = Path(file_path)

            if not path.exists():
                return {
                    "success": False,
                    "error": "not_found",
                    "message": f"Arquivo não encontrado: {file_path}",
                }

            if not path.is_file():
                return {
                    "success": False,
                    "error": "not_a_file",
                    "message": f"O caminho não é um arquivo: {file_path}",
                }

            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            return {
                "success": True,
                "content": content,
                "file_path": str(path.absolute()),
                "size": path.stat().st_size,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao ler arquivo: {str(e)}",
            }

    @staticmethod
    def write(
        file_path: str, content: str, append: bool = False, encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """
        Escreve conteúdo em um arquivo.

        Args:
            file_path: Caminho do arquivo
            content: Conteúdo a ser escrito
            append: Se deve adicionar ao final do arquivo
            encoding: Codificação do arquivo

        Returns:
            Dicionário com o resultado da operação
        """
        try:
            path = Path(file_path)

            # Criar diretórios se necessário
            os.makedirs(path.parent, exist_ok=True)

            # Modo de abertura
            mode = "a" if append else "w"

            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)

            return {
                "success": True,
                "file_path": str(path.absolute()),
                "append": append,
                "size": path.stat().st_size,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao escrever arquivo: {str(e)}",
            }

    @staticmethod
    def delete(file_path: str) -> Dict[str, Any]:
        """
        Remove um arquivo.

        Args:
            file_path: Caminho do arquivo

        Returns:
            Dicionário com o resultado da operação
        """
        try:
            path = Path(file_path)

            if not path.exists():
                return {
                    "success": False,
                    "error": "not_found",
                    "message": f"Arquivo não encontrado: {file_path}",
                }

            if not path.is_file():
                return {
                    "success": False,
                    "error": "not_a_file",
                    "message": f"O caminho não é um arquivo: {file_path}",
                }

            # Salvar informações antes de remover
            file_info = {
                "path": str(path.absolute()),
                "size": path.stat().st_size,
                "exists": True,
            }

            # Remover o arquivo
            path.unlink()

            return {
                "success": True,
                "file_info": file_info,
                "message": f"Arquivo removido: {file_path}",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao remover arquivo: {str(e)}",
            }

    @staticmethod
    def list_dir(dir_path: str) -> Dict[str, Any]:
        """
        Lista o conteúdo de um diretório.

        Args:
            dir_path: Caminho do diretório

        Returns:
            Dicionário com o resultado da operação
        """
        try:
            path = Path(dir_path)

            if not path.exists():
                return {
                    "success": False,
                    "error": "not_found",
                    "message": f"Diretório não encontrado: {dir_path}",
                }

            if not path.is_dir():
                return {
                    "success": False,
                    "error": "not_a_directory",
                    "message": f"O caminho não é um diretório: {dir_path}",
                }

            # Listar o conteúdo
            contents = list(path.iterdir())

            # Separar arquivos e diretórios
            files = [str(p.name) for p in contents if p.is_file()]
            dirs = [str(p.name) for p in contents if p.is_dir()]

            return {
                "success": True,
                "directory": str(path.absolute()),
                "files": files,
                "directories": dirs,
                "count": {
                    "files": len(files),
                    "directories": len(dirs),
                    "total": len(contents),
                },
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao listar diretório: {str(e)}",
            }
