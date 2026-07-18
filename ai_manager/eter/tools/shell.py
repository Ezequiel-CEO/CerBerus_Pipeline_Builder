#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ferramentas de shell para o assistente Èter.

Este módulo fornece funcionalidades para interagir com o sistema operacional,
executar comandos e gerenciar processos.
"""

import os
import sys
import signal
import platform
import subprocess
import shlex
from typing import Dict, List, Any, Optional, Union, Tuple

from cerberus_api.utils.logging_config import get_logger

logger = get_logger("eter_shell")


class ShellCommands:
    """
    Classe para executar comandos de shell e gerenciar processos.
    """

    @staticmethod
    def execute(
        command: str,
        cwd: Optional[str] = None,
        capture_output: bool = True,
        timeout: Optional[int] = None,
        shell: bool = True,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Executa um comando no shell e retorna o resultado.

        Args:
            command: Comando a ser executado
            cwd: Diretório de trabalho
            capture_output: Se deve capturar saídas stdout/stderr
            timeout: Tempo limite de execução em segundos
            shell: Se deve usar shell para executar o comando
            env: Variáveis de ambiente adicionais

        Returns:
            Dicionário com o resultado da execução
        """
        logger.info(f"Executando comando: {command}")

        try:
            # Preparar ambiente
            exec_env = os.environ.copy()
            if env:
                exec_env.update(env)

            # Executar o comando
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=capture_output,
                text=True,
                cwd=cwd,
                timeout=timeout,
                env=exec_env,
            )

            # Registrar resultado
            logger.debug(f"Comando finalizado com código: {result.returncode}")

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout if capture_output else None,
                "stderr": result.stderr if capture_output else None,
                "command": command,
            }

        except subprocess.TimeoutExpired:
            logger.warning(f"Comando excedeu o tempo limite: {command}")
            return {
                "success": False,
                "error": "timeout",
                "message": f"Comando excedeu o tempo limite de {timeout}s",
                "command": command,
            }

        except Exception as e:
            logger.error(f"Erro ao executar comando '{command}': {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao executar comando: {str(e)}",
                "command": command,
            }

    @staticmethod
    def execute_background(
        command: str,
        cwd: Optional[str] = None,
        output_file: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Executa um comando em segundo plano (não bloqueante).

        Args:
            command: Comando a ser executado
            cwd: Diretório de trabalho
            output_file: Arquivo para redirecionar saída
            env: Variáveis de ambiente adicionais

        Returns:
            Dicionário com informações sobre o processo iniciado
        """
        logger.info(f"Iniciando comando em segundo plano: {command}")

        try:
            # Preparar ambiente
            exec_env = os.environ.copy()
            if env:
                exec_env.update(env)

            # Redirecionar saída se necessário
            if output_file:
                output_fd = open(output_file, "w")
            else:
                output_fd = subprocess.DEVNULL

            # Iniciar o processo
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=output_fd,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=exec_env,
            )

            # Armazenar informações do processo
            info = {
                "success": True,
                "pid": process.pid,
                "command": command,
                "output_file": output_file,
                "status": "running",
            }

            logger.debug(f"Processo iniciado em segundo plano: PID={process.pid}")

            return info

        except Exception as e:
            logger.error(f"Erro ao iniciar comando em segundo plano '{command}': {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao iniciar comando: {str(e)}",
                "command": command,
            }

    @staticmethod
    def kill_process(pid: int, force: bool = False) -> Dict[str, Any]:
        """
        Encerra um processo pelo PID.

        Args:
            pid: ID do processo a ser encerrado
            force: Se True, força o encerramento (SIGKILL)

        Returns:
            Dicionário com o resultado da operação
        """
        logger.info(f"Tentando encerrar processo: PID={pid}, force={force}")

        try:
            # Verificar se o processo existe
            if not ShellCommands._pid_exists(pid):
                return {
                    "success": False,
                    "error": "process_not_found",
                    "message": f"Processo com PID {pid} não encontrado",
                }

            # Sinal a ser enviado
            sig = signal.SIGKILL if force else signal.SIGTERM

            # Enviar sinal para encerrar
            os.kill(pid, sig)

            return {
                "success": True,
                "pid": pid,
                "force": force,
                "message": f"Sinal {'SIGKILL' if force else 'SIGTERM'} enviado para o processo {pid}",
            }

        except Exception as e:
            logger.error(f"Erro ao encerrar processo {pid}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao encerrar processo: {str(e)}",
            }

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Obtém informações sobre o sistema operacional.

        Returns:
            Dicionário com informações do sistema
        """
        try:
            # Coletar informações básicas
            info = {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python": sys.version,
                "hostname": platform.node(),
                "username": os.getlogin() if hasattr(os, "getlogin") else None,
            }

            # Adicionar variáveis de ambiente relevantes
            info["env"] = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
                "USER": os.environ.get("USER", ""),
                "SHELL": os.environ.get("SHELL", ""),
                "PWD": os.environ.get("PWD", ""),
            }

            return {"success": True, "info": info}

        except Exception as e:
            logger.error(f"Erro ao obter informações do sistema: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao obter informações do sistema: {str(e)}",
            }

    @staticmethod
    def find_command_path(command: str) -> Dict[str, Any]:
        """
        Encontra o caminho completo de um comando no sistema.

        Args:
            command: Nome do comando a ser localizado

        Returns:
            Dicionário com o resultado da busca
        """
        try:
            # Em sistemas Unix-like, usar o comando 'which'
            if os.name == "posix":
                result = subprocess.run(
                    f"which {shlex.quote(command)}",
                    shell=True,
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    path = result.stdout.strip()
                    return {
                        "success": True,
                        "command": command,
                        "path": path,
                        "exists": True,
                    }
                else:
                    return {
                        "success": True,
                        "command": command,
                        "exists": False,
                        "message": f"Comando '{command}' não encontrado no PATH",
                    }

            # Em sistemas Windows, verificar extensões comuns
            elif os.name == "nt":
                extensions = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";")

                for directory in os.environ.get("PATH", "").split(os.pathsep):
                    for ext in extensions:
                        path = os.path.join(directory, command + ext)
                        if os.path.isfile(path):
                            return {
                                "success": True,
                                "command": command,
                                "path": path,
                                "exists": True,
                            }

                return {
                    "success": True,
                    "command": command,
                    "exists": False,
                    "message": f"Comando '{command}' não encontrado no PATH",
                }

            # Caso não seja possível determinar
            else:
                return {
                    "success": False,
                    "error": "unsupported_platform",
                    "message": f"Plataforma não suportada para localizar comandos: {os.name}",
                }

        except Exception as e:
            logger.error(f"Erro ao localizar caminho do comando '{command}': {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Erro ao localizar caminho do comando: {str(e)}",
            }

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        """
        Verifica se um processo com o PID especificado existe.

        Args:
            pid: ID do processo

        Returns:
            True se o processo existir, False caso contrário
        """
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


# Função auxiliar para uso direto do módulo
def execute_command(command: str, **kwargs) -> Dict[str, Any]:
    """
    Wrapper para execução rápida de comandos.

    Args:
        command: Comando a ser executado
        **kwargs: Argumentos adicionais para ShellCommands.execute

    Returns:
        Resultado da execução do comando
    """
    return ShellCommands.execute(command, **kwargs)
