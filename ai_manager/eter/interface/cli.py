#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface de linha de comando para o Èter

Este módulo implementa uma interface de linha de comando para interagir
com o assistente Èter. Permite que usuários enviem comandos, façam
consultas e naveguem pelo conhecimento armazenado.
"""

import os
import sys
import cmd
import time
import argparse
import textwrap
from typing import Optional, List, Dict, Any
import logging
from colorama import Fore, Style

from cerberus_api.pipeline_builder.ai_manager.eter.agent import (
    EterAgent,
    GGUF_AVAILABLE,
)
from cerberus_api.utils.logging_config import get_logger
from cerberus_api.pipeline_builder.ai_manager.eter.security.access_control import AccessControl


class EterCLI(cmd.Cmd):
    """
    Interface de linha de comando para interagir com o Èter.

    Implementa uma shell interativa que processa comandos do usuário
    e os encaminha para o agente Èter, exibindo as respostas de forma
    amigável.
    """

    intro = """
    =====================================================
     Èter - Assistente Inteligente do CerBerus FMK
    =====================================================
     Digite 'ajuda' para ver os comandos disponíveis
     Digite 'sair' para encerrar
    =====================================================
    """

    prompt = "\nÈter> "

    def __init__(self, agent: EterAgent, verbose: bool = False, config_dir: str = None):
        """
        Inicializa a interface de linha de comando.

        Args:
            agent: Instância do agente Èter
            verbose: Se True, exibe logs detalhados
            config_dir: Diretório de configuração para AccessControl
        """
        super().__init__()
        self.agent = agent
        self.verbose = verbose
        self.logger = get_logger("eter_cli")

        # Status do streaming de LLM
        self.streaming_enabled = False

        # Inicializar controle de acesso
        if config_dir is None:
            config_dir = os.path.join(os.getcwd(), "eter_data", "security")
        self.access_control = AccessControl(
            config_dir=config_dir,
            user_id=getattr(agent, 'user_id', 'default_user'),
            verbose=verbose
        )

        self.logger.info("CLI inicializada com AccessControl")

        # Exibir status do GGUF/LLM
        if GGUF_AVAILABLE:
            print("✓ Suporte a modelos GGUF está disponível")
            models = self.agent.list_available_models()
            if models:
                print(f"✓ {len(models)} modelos encontrados")
                if self.agent.active_model is not None:
                    print(f"✓ Modelo ativo: {self.agent.active_model.model_path}")
                else:
                    print("⚠ Nenhum modelo ativo")
            else:
                print("⚠ Nenhum modelo GGUF encontrado no diretório de modelos")
        else:
            print("⚠ Suporte a modelos GGUF não está disponível")

        # Exibir permissões do usuário
        self._print_user_permissions()

    def _print_user_permissions(self):
        """Exibe as permissões do usuário atual."""
        try:
            perms = self.access_control.get_user_permissions()
            print(f"\n👤 Usuário: {perms['user_id']}")
            print(f"🔑 Nível: {perms['permission_level']}")
            print(f"✓ Operações permitidas: {', '.join(perms['allowed_operations'][:5])}{'...' if len(perms['allowed_operations']) > 5 else ''}")
            print()
        except Exception as e:
            self.logger.error(f"Erro ao obter permissões: {e}")

    def _check_permission(self, operation: str) -> bool:
        """
        Verifica se o usuário tem permissão para uma operação.

        Args:
            operation: Nome da operação

        Returns:
            True se permitido, False caso contrário
        """
        allowed, message = self.access_control.validate_operation(operation, {})
        if not allowed:
            print(f"\n{Fore.RED}⛔ Acesso negado: {message}{Style.RESET_ALL}")
            self.logger.warning(f"Acesso negado para {self.access_control.user_id} em {operation}")
        return allowed

    def emptyline(self):
        """Não faz nada ao pressionar Enter sem comando."""
        pass

    def default(self, line: str):
        """
        Processa entrada não reconhecida como comando como uma consulta ao agente.

        Args:
            line: Texto de entrada do usuário
        """
        if not line.strip():
            return

        # Tratar como consulta ao agente (não como comando shell)
        self._process_command(line)

    def _print_wrapped_response(self, text: str):
        """
        Imprime texto com quebra de linha em largura apropriada.

        Args:
            text: Texto a ser exibido
        """
        print("\nÈter:")
        # Obter largura do terminal
        try:
            terminal_width = os.get_terminal_size().columns
        except:
            terminal_width = 80

        # Ajustar para margens e comprimento razoável
        width = min(terminal_width - 2, 100)

        for line in text.split("\n"):
            if line.strip():
                wrapped = textwrap.fill(line, width=width)
                print(wrapped)
            else:
                print()
        print()

    def do_ajuda(self, arg: str):
        """
        Exibe a ajuda sobre os comandos disponíveis.

        Uso:
          ajuda              - Lista todos os comandos
          ajuda <comando>    - Exibe ajuda específica para um comando
        """
        if not arg:
            print("\nComandos disponíveis:")
            print("  ajuda       - Mostra esta mensagem de ajuda")
            print("  sair        - Encerra a sessão")
            print("  limpar      - Limpa a tela do terminal")
            print("  modelos     - Gerencia modelos GGUF (listar, carregar, ativo)")
            print("  streaming   - Controla exibição em tempo real")
            print("  conversa    - Gerencia histórico de conversas")
            print("  executar    - Executa um comando no shell")
            print("  arquivo     - Gerencia operações de arquivos")
            print("  dir         - Lista o conteúdo de um diretório")
            print("  status      - Status do sistema e framework")
            print("\nDigite qualquer texto para interagir com o Èter.")
            return

        # Mapear nomes de comandos para suas descrições
        if arg == "modelos":
            print("\nComando: modelos")
            print("  Gerencia os modelos de linguagem GGUF disponíveis.")
            print("\nUso:")
            print("  modelos listar    - Lista os modelos disponíveis")
            print("  modelos carregar <nome>  - Carrega um modelo específico")
            print("  modelos ativo     - Mostra informações do modelo ativo")
            return

        if arg == "streaming":
            print("\nComando: streaming")
            print("  Ativa ou desativa a exibição de respostas em tempo real.")
            print(
                "  Quando ativado, você verá as respostas sendo geradas caractere por caractere."
            )
            print("\nUso:")
            print("  streaming on    - Ativa o streaming")
            print("  streaming off   - Desativa o streaming")
            print("  streaming       - Exibe o status atual")
            return

        if arg == "conversa":
            print("\nComando: conversa")
            print("  Gerencia o histórico de conversas com o assistente.")
            print("\nUso:")
            print("  conversa mostrar [n] - Mostra últimas n mensagens (padrão: 10)")
            print("  conversa limpar     - Limpa histórico de conversa")
            return

        if arg == "executar":
            print("\nComando: executar")
            print("  Executa um comando no shell do sistema operacional.")
            print("\nUso:")
            print("  executar <comando>  - Executa o comando especificado")
            print("\nExemplos:")
            print("  executar ls -la")
            print("  executar python -c 'print(\"Hello World\")'")
            return

        if arg == "arquivo":
            print("\nComando: arquivo")
            print("  Gerencia operações de leitura, escrita e manipulação de arquivos.")
            print("\nUso:")
            print("  arquivo ler <caminho>     - Lê e exibe o conteúdo de um arquivo")
            print(
                "  arquivo criar <caminho>   - Cria um novo arquivo (pedirá o conteúdo)"
            )
            print("  arquivo editar <caminho>  - Edita um arquivo existente")
            print("  arquivo deletar <caminho> - Remove um arquivo")
            return

        if arg == "dir":
            print("\nComando: dir")
            print("  Lista o conteúdo de um diretório.")
            print("\nUso:")
            print(
                "  dir [caminho]  - Lista o conteúdo do diretório (padrão: diretório atual)"
            )
            return

        if arg == "status":
            print("\nComando: status")
            print("  Exibe status do sistema e do framework.")
            print("\nUso:")
            print("  status            - Status completo")
            print("  status sistema    - Apenas sistema operacional")
            print("  status framework  - Apenas framework CerBerus")
            return

        # Comando não reconhecido
        print(
            f"Comando '{arg}' não encontrado. Digite 'ajuda' para ver a lista de comandos."
        )

    def do_sair(self, arg: str):
        """
        Encerra a sessão com o Èter.

        Uso: sair
        """
        print("Encerrando sessão...")
        self.agent.save_state()
        return True

    def do_EOF(self, arg: str):
        """Trata Ctrl+D como comando de saída."""
        print("\nEncerrando sessão...")
        return self.do_sair("")

    def do_limpar(self, arg: str):
        """
        Limpa a tela do terminal.

        Uso: limpar
        """
        os.system("cls" if os.name == "nt" else "clear")

    def do_modelos(self, arg: str):
        """
        Gerencia modelos GGUF.

        Uso:
          modelos listar - Lista todos os modelos disponíveis
          modelos carregar <nome> - Carrega um modelo específico
          modelos ativo - Mostra o modelo atualmente ativo
        """
        if not GGUF_AVAILABLE:
            print("⚠ Suporte a modelos GGUF não está disponível.")
            print(
                "  Instale o pacote 'llama-cpp-python' para habilitar esta funcionalidade."
            )
            return

        args = arg.strip().split()

        # Sem argumentos, mostrar instrução de uso
        if not args:
            print("Uso: modelos [listar|carregar|ativo]")
            return

        command = args[0].lower()

        # Verificar permissões
        if command == "listar":
            if not self._check_permission("query"):
                return
        elif command == "carregar":
            if not self._check_permission("install_components"):
                return
        elif command == "ativo":
            if not self._check_permission("query"):
                return

        # Listar modelos disponíveis
        if command == "listar":
            models = self.agent.list_available_models()

            if not models:
                print("Nenhum modelo encontrado no diretório de modelos.")
                print(f"Diretório de modelos: {self.agent.model_dir}")
                return

            print(f"\nModelos disponíveis ({len(models)}):")
            for i, model in enumerate(models, 1):
                # Verificar se o modelo tem as chaves esperadas
                model_name = model.get("name", "Desconhecido")
                model_path = model.get("path", "")
                model_size = model.get("size", 0)

                # Calcular tamanho em GB
                size_gb = (
                    model_size / (1024**3)
                    if isinstance(model_size, (int, float))
                    else 0
                )

                # Verificar se é o modelo ativo
                is_active = False
                if self.agent.active_model is not None:
                    active_path = getattr(self.agent.active_model, "model_path", "")
                    is_active = model_path == active_path

                active_mark = " (ATIVO)" if is_active else ""
                print(f"{i}. {model_name} - {size_gb:.1f}GB{active_mark}")

            print("\nPara carregar um modelo: modelos carregar <nome>")
            return

        # Carregar um modelo específico
        elif command == "carregar":
            if len(args) < 2:
                print("Erro: Especifique o nome do modelo para carregar.")
                print("Uso: modelos carregar <nome>")
                return

            model_name = " ".join(args[1:])

            # Verificar modelos disponíveis
            try:
                models = self.agent.list_available_models()

                # Verificar correspondências pelo nome
                matching_models = []
                for model in models:
                    if isinstance(model, dict) and "name" in model:
                        if model_name.lower() in model["name"].lower():
                            matching_models.append(model)

                if not matching_models:
                    print(f"Modelo '{model_name}' não encontrado.")
                    print("Use 'modelos listar' para ver os modelos disponíveis.")
                    return

                # Se temos múltiplas correspondências, usar a primeira
                target_model = matching_models[0].get("name", "")
                if len(matching_models) > 1:
                    print(f"Múltiplos modelos correspondem a '{model_name}'.")
                    print(f"Carregando o primeiro correspondente: '{target_model}'")

                print(f"Carregando modelo '{target_model}'...")
                success = self.agent.set_active_model(target_model)

                if success:
                    print(f"✓ Modelo '{target_model}' carregado com sucesso!")
                else:
                    print(f"✗ Falha ao carregar modelo '{target_model}'.")

            except Exception as e:
                print(f"Erro ao processar o comando: {str(e)}")
                self.logger.error(f"Erro ao carregar modelo: {str(e)}")

            return

        # Mostrar modelo ativo
        elif command == "ativo":
            if self.agent.active_model is None:
                print("Nenhum modelo está ativo no momento.")
                print("Use 'modelos carregar <nome>' para ativar um modelo.")
                return

            # Exibir informações do modelo ativo
            model_info = {
                "nome": os.path.basename(self.agent.active_model.model_path),
                "caminho": self.agent.active_model.model_path,
                "memória": f"{self.agent.active_model.get_memory_usage():.1f}MB",
            }

            print("\nModelo ativo:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")

            return

        # Comando não reconhecido
        else:
            print(f"Comando de modelo não reconhecido: '{command}'")
            print("Comandos disponíveis: listar, carregar, ativo")

    def do_streaming(self, arg: str):
        """
        Ativa/desativa streaming de respostas em tempo real.

        Uso:
          streaming on  - Ativa o streaming
          streaming off - Desativa o streaming
          streaming     - Exibe o status atual
        """
        if not self._check_permission("execute"):
            return

        if not GGUF_AVAILABLE or self.agent.active_model is None:
            print("⚠ Streaming requer um modelo GGUF ativo.")
            print("  Use 'modelos carregar <nome>' para ativar um modelo.")
            return

        arg = arg.strip().lower()

        # Exibir status atual
        if not arg:
            status = "ativado" if self.streaming_enabled else "desativado"
            print(f"Streaming está {status}.")
            print("Use 'streaming on' ou 'streaming off' para alterar.")
            return

        # Ativar streaming
        if arg == "on":
            self.streaming_enabled = True
            print("✓ Streaming ativado. Respostas serão exibidas em tempo real.")

        # Desativar streaming
        elif arg == "off":
            self.streaming_enabled = False
            print("✓ Streaming desativado. Respostas serão exibidas apenas ao final.")

        # Comando inválido
        else:
            print(f"Opção inválida: '{arg}'")
            print("Use 'streaming on' ou 'streaming off'.")

    def do_conversa(self, arg: str):
        """
        Gerencia o histórico de conversas.

        Uso:
          conversa mostrar [n] - Mostra as últimas n mensagens (padrão: 10)
          conversa limpar     - Limpa o histórico de conversa
        """
        args = arg.strip().split()

        # Sem argumentos, mostrar ajuda do comando
        if not args:
            print("Uso: conversa [mostrar|limpar]")
            return

        command = args[0].lower()

        # Mostrar histórico de conversa
        if command == "mostrar":
            if not self._check_permission("query"):
                return
            # Determinar quantas mensagens mostrar
            limit = 10  # Padrão
            if len(args) > 1:
                try:
                    limit = int(args[1])
                except ValueError:
                    print(f"Erro: '{args[1]}' não é um número válido.")
                    return

            # Obter histórico de conversa
            history = self.agent.memory.get_conversation_history(limit=limit)

            if not history:
                print("Histórico de conversa vazio.")
                return

            print(
                f"\nHistórico de conversa (últimas {min(limit, len(history))} mensagens):"
            )

            for i, entry in enumerate(history, 1):
                role = entry.get("role", "")
                content = entry.get("content", "")
                timestamp = entry.get("timestamp", "")

                # Formatar timestamp
                if timestamp:
                    try:
                        # Converter de ISO para formato mais legível
                        t = time.strptime(timestamp.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                        formatted_time = time.strftime("%d/%m/%Y %H:%M:%S", t)
                    except:
                        formatted_time = timestamp
                else:
                    formatted_time = "desconhecido"

                # Exibir entrada formatada
                if role == "user":
                    print(f"\n[{i}] Usuário ({formatted_time}):")
                    print(f"  {content}")
                elif role == "assistant":
                    print(f"\n[{i}] Èter ({formatted_time}):")
                    print(f"  {content}")
                else:
                    print(f"\n[{i}] {role} ({formatted_time}):")
                    print(f"  {content}")

            print("\n")
            return

        # Limpar histórico de conversa
        elif command == "limpar":
            if not self._check_permission("save_data"):
                return
            # TODO: Implementar limpeza de histórico
            print("Funcionalidade de limpar histórico ainda não implementada.")
            return

        # Comando não reconhecido
        else:
            print(f"Comando de conversa não reconhecido: '{command}'")
            print("Comandos disponíveis: mostrar, limpar")

    def do_executar(self, arg: str):
        """
        Executa um comando no shell do sistema operacional.

        Uso: executar <comando>

        Exemplos:
          executar ls -la
          executar python -c 'print("Hello World")'
        """
        if not self._check_permission("execute_command"):
            return

        if not arg:
            print("Erro: Especifique um comando para executar.")
            print("Uso: executar <comando>")
            return

        print(f"Executando: {arg}")
        result = self.agent.execute_shell_command(arg)

        # Exibir resultado
        if result["status"] == "success":
            print(
                f"\nComando executado com sucesso (código {result['data']['returncode']})"
            )

            # Exibir saída stdout se houver
            if result["data"]["stdout"]:
                print("\nSaída:")
                print(result["data"]["stdout"])

            # Exibir stderr se houver
            if result["data"]["stderr"]:
                print("\nErros/Avisos:")
                print(result["data"]["stderr"])

        else:
            print(f"\nErro ao executar comando: {result['message']}")

    def do_arquivo(self, arg: str):
        """
        Gerencia operações de leitura, escrita e manipulação de arquivos.

        Uso:
          arquivo ler <caminho>     - Lê e exibe o conteúdo de um arquivo
          arquivo criar <caminho>   - Cria um novo arquivo (pedirá o conteúdo)
          arquivo editar <caminho>  - Edita um arquivo existente
          arquivo deletar <caminho> - Remove um arquivo
        """
        args = arg.strip().split()

        if not args:
            print("Erro: Especifique uma operação de arquivo.")
            print("Uso: arquivo [ler|criar|editar|deletar] <caminho>")
            return

        operation = args[0].lower()

        # Verificar permissões por operação
        if operation == "ler":
            if not self._check_permission("read"):
                return
        elif operation in ("criar", "editar"):
            if not self._check_permission("write"):
                return
        elif operation == "deletar":
            if not self._check_permission("delete_file"):
                return

        # Verificar se temos caminho do arquivo
        if len(args) < 2:
            print(
                f"Erro: Especifique o caminho do arquivo para a operação '{operation}'."
            )
            return

        # Obter caminho do arquivo
        file_path = " ".join(args[1:])

        # Operação de leitura
        if operation == "ler":
            result = self.agent.read_file(file_path)

            if result["status"] == "success":
                print(f"\nConteúdo de '{file_path}':\n")
                print(result["data"]["content"])
                print("\n")
            else:
                print(f"\nErro ao ler arquivo: {result['message']}")

            return

        # Operação de criação
        elif operation == "criar":
            print(f"\nCriando novo arquivo: {file_path}")
            print("Digite o conteúdo (finalize com Ctrl+D em uma nova linha):")

            # Coletar conteúdo do usuário
            content_lines = []
            try:
                while True:
                    line = input()
                    content_lines.append(line)
            except EOFError:
                pass

            content = "\n".join(content_lines)

            # Criar o arquivo
            result = self.agent.write_file(file_path, content)

            if result["status"] == "success":
                print(f"\nArquivo '{file_path}' criado com sucesso.")
            else:
                print(f"\nErro ao criar arquivo: {result['message']}")

            return

        # Operação de edição
        elif operation == "editar":
            # Primeiro ler o arquivo
            read_result = self.agent.read_file(file_path)

            if read_result["status"] != "success":
                print(f"\nErro ao ler arquivo para edição: {read_result['message']}")
                return

            # Exibir conteúdo atual e instruções
            print(f"\nEditando arquivo: {file_path}")
            print("Conteúdo atual:")
            print(read_result["data"]["content"])
            print("\nDigite o novo conteúdo (finalize com Ctrl+D em uma nova linha):")

            # Coletar novo conteúdo
            content_lines = []
            try:
                while True:
                    line = input()
                    content_lines.append(line)
            except EOFError:
                pass

            content = "\n".join(content_lines)

            # Confirmar
            confirmation = input(f"\nSobrescrever o arquivo '{file_path}'? (s/n): ")
            if confirmation.lower() not in ["s", "sim", "y", "yes"]:
                print("Operação cancelada.")
                return

            # Salvar alterações
            write_result = self.agent.write_file(file_path, content)

            if write_result["status"] == "success":
                print(f"\nArquivo '{file_path}' atualizado com sucesso.")
            else:
                print(f"\nErro ao atualizar arquivo: {write_result['message']}")

            return

        # Operação de deleção
        elif operation == "deletar":
            # Confirmar primeiro
            confirmation = input(
                f"\nTem certeza que deseja deletar o arquivo '{file_path}'? (s/n): "
            )
            if confirmation.lower() not in ["s", "sim", "y", "yes"]:
                print("Operação cancelada.")
                return

            # Remover o arquivo
            result = self.agent.delete_file(file_path)

            if result["status"] == "success":
                print(f"\nArquivo '{file_path}' removido com sucesso.")
            else:
                print(f"\nErro ao remover arquivo: {result['message']}")

            return

        # Operação não reconhecida
        else:
            print(f"Operação de arquivo não reconhecida: '{operation}'")
            print("Operações disponíveis: ler, criar, editar, deletar")
            return

    def do_dir(self, arg: str):
        """
        Lista o conteúdo de um diretório.

        Uso: dir [caminho]

        Se o caminho não for especificado, lista o diretório atual.
        """
        if not self._check_permission("list_dir"):
            return

        # Usar diretório atual se não especificado
        dir_path = arg.strip() if arg.strip() else "."

        result = self.agent.list_directory(dir_path)

        if result["status"] == "success":
            print(f"\nConteúdo do diretório: {result['data']['directory']}")

            # Exibir subdiretórios
            if result["data"]["directories"]:
                print("\nDiretórios:")
                for directory in sorted(result["data"]["directories"]):
                    print(f"  📁 {directory}")

            # Exibir arquivos
            if result["data"]["files"]:
                print("\nArquivos:")
                for file in sorted(result["data"]["files"]):
                    print(f"  📄 {file}")

            # Se estiver vazio
            if not result["data"]["all_items"]:
                print("\nDiretório vazio.")

        else:
            print(f"\nErro ao listar diretório: {result['message']}")

    def do_status(self, arg: str):
        """
        Exibe status detalhado do sistema e do framework.

        Uso:
          status    - Status completo (CPU, GPU, memória, framework)
          status sistema    - Apenas informações do sistema operacional
          status framework  - Apenas status do framework CerBerus
        """
        args = arg.strip().split() if arg.strip() else []
        sub = args[0].lower() if args else "completo"

        if sub == "sistema":
            if not self._check_permission("query"):
                return
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage("/")
                print(f"\n🖥️  Sistema Operacional:")
                print(f"  CPU: {cpu_percent}%")
                print(f"  Memória: {memory.percent}% ( {memory.used // (1024**2)}MB / {memory.total // (1024**2)}MB )")
                print(f"  Disco: {disk.percent}% ( {disk.free // (1024**3)}GB livres de {disk.total // (1024**3)}GB )")
            except Exception as e:
                print(f"Erro ao obter status do sistema: {e}")

        elif sub == "framework":
            if not self._check_permission("query"):
                return
            try:
                framework = self.agent.framework
                if framework and hasattr(framework, 'list_available_components'):
                    components = framework.list_available_components()
                    print(f"\n🔧 Framework CerBerusFMK:")
                    print(f"  Módulos: {len(components.get('modules', []))}")
                    print(f"  Classes: {len(components.get('classes', []))}")
                    print(f"  Funções: {len(components.get('functions', []))}")
                    print(f"  APIs: {len(components.get('apis', []))}")
                else:
                    print("Framework não disponível.")
            except Exception as e:
                print(f"Erro ao obter status do framework: {e}")

        else:
            # Status completo
            if not self._check_permission("query"):
                return
            self.do_status("sistema")
            self.do_status("framework")
            try:
                status = self.agent.framework.get_status() if self.agent.framework else {}
                if status:
                    print(f"\n📊 Status do Èter:")
                    for key, value in status.items():
                        print(f"  {key}: {value}")
            except Exception as e:
                print(f"Erro ao obter status do Èter: {e}")

    def _process_command(self, command: str) -> None:
        """
        Processa um comando de usuário e exibe a resposta.

        Args:
            command: O comando a ser processado.
        """
        if not command.strip():
            return

        self.logger.info(f"Processando comando: {command}")

        # Verificar permissão para interagir com o agente (execute)
        if not self._check_permission("execute"):
            return

        # Para comandos normais, usar o modelo ativo para processar
        try:
            if not self.agent.is_model_active():
                print(
                    f"\n{Fore.YELLOW}Atenção: Nenhum modelo ativo. Usando respostas simuladas.{Style.RESET_ALL}"
                )

            start_time = time.time()
            response = self.agent.process_input(command)
            elapsed_time = time.time() - start_time

            print(
                f"\n{Fore.CYAN}Resposta do Èter ({elapsed_time:.2f}s):{Style.RESET_ALL}"
            )
            print(f"\n{response}")

        except Exception as e:
            self.logger.error(f"Erro ao processar comando: {e}")
            print(f"\n{Fore.RED}Erro ao processar comando: {e}{Style.RESET_ALL}")


def initialize(
    models_dir: str = "./models",
    knowledge_dir: str = "./knowledge",
    workspace_dir: str = "./workspace",
    user_id: str = "default_user",
    verbose: bool = False,
) -> EterCLI:
    """
    Inicializa o ambiente do Èter e retorna a CLI.

    Args:
        models_dir: Diretório para armazenar modelos
        knowledge_dir: Diretório para armazenar conhecimento
        workspace_dir: Diretório do workspace atual
        user_id: ID do usuário atual
        verbose: Se True, exibe logs detalhados

    Returns:
        Interface de linha de comando inicializada
    """
    # Garantir que diretórios existam
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(knowledge_dir, exist_ok=True)
    os.makedirs(workspace_dir, exist_ok=True)

    # Diretório de segurança para AccessControl
    security_dir = os.path.join(workspace_dir, "security")
    os.makedirs(security_dir, exist_ok=True)

    # Inicializar o agente
    agent = EterAgent(
        model_dir=models_dir,
        knowledge_dir=knowledge_dir,
        workspace_dir=workspace_dir,
        user_id=user_id,
        verbose=verbose,
    )

    # Inicializar CLI com controle de acesso
    cli = EterCLI(agent, verbose=verbose, config_dir=security_dir)

    return cli


def main():
    """Função principal para iniciar o CLI do assistente Èter."""
    # Configurar argumentos da linha de comando
    parser = argparse.ArgumentParser(
        description="Èter - Assistente Inteligente CerBerus"
    )

    parser.add_argument(
        "--models-dir", default="./models", help="Diretório para armazenar modelos"
    )
    parser.add_argument(
        "--knowledge-dir",
        default="./knowledge",
        help="Diretório para armazenar conhecimento",
    )
    parser.add_argument(
        "--workspace-dir", default="./workspace", help="Diretório do workspace atual"
    )
    parser.add_argument(
        "--user-id", default=os.getenv("USER", "default_user"), help="ID do usuário"
    )
    parser.add_argument("--verbose", action="store_true", help="Exibir logs detalhados")
    parser.add_argument(
        "--auto-load", action="store_true", help="Carregar modelo automaticamente"
    )
    parser.add_argument(
        "--model",
        default="Qwen2.5-7B-Instruct-1M-Q6_K",
        help="Nome do modelo a carregar automaticamente",
    )
    parser.add_argument(
        "--stream", action="store_true", help="Ativar streaming de respostas"
    )

    args = parser.parse_args()

    # Configurar logging
    log_level = logging.DEBUG if args.verbose else logging.INFO

    try:
        from cerberus_api.utils.logging_config import get_logger

        logger = get_logger("eter_cli", level=log_level)
    except ImportError:
        # Configuração de fallback
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger("eter_cli")

    cli = None

    try:
        # Inicializar CLI
        cli = initialize(
            models_dir=args.models_dir,
            knowledge_dir=args.knowledge_dir,
            workspace_dir=args.workspace_dir,
            user_id=args.user_id,
            verbose=args.verbose,
        )

        # Carregar modelo automaticamente se solicitado
        if args.auto_load and cli and cli.agent:
            try:
                print(f"Carregando modelo automaticamente: {args.model}")
                # Adicionar caminho completo do modelo
                model_path = os.path.expanduser(
                    f"~/.cerberusfmk/models/{args.model}.gguf"
                )
                if os.path.exists(model_path):
                    try:
                        # Verificar se o ModelHub está acessível e tem o método add_model
                        if (
                            hasattr(cli.agent, "llm_manager")
                            and cli.agent.llm_manager
                            and hasattr(cli.agent.llm_manager, "add_model")
                        ):
                            # Dividir o nome para remover a extensão se presente
                            model_name = args.model.split(".")[0]
                            # Registrar o modelo no catálogo antes de tentar carregá-lo
                            cli.agent.llm_manager.add_model(model_name, model_path)
                            # Configurar como modelo ativo
                            cli.agent.set_active_model(model_name)
                            logger.info(
                                f"Modelo '{model_name}' carregado automaticamente"
                            )
                        else:
                            logger.warning(
                                "ModelHub não tem método add_model ou não está disponível"
                            )
                            print("⚠ ModelHub não está configurado corretamente.")
                            print(
                                "  O Èter continuará funcionando, mas sem recursos avançados de IA."
                            )
                    except Exception as model_error:
                        logger.error(f"Erro ao carregar modelo: {str(model_error)}")
                        print(f"⚠ Erro ao carregar modelo: {str(model_error)}")
                        print(
                            "  O Èter continuará funcionando, mas sem recursos avançados de IA."
                        )
                else:
                    logger.error(f"Modelo não encontrado: {model_path}")
                    print(f"⚠ Modelo não encontrado: {model_path}")
                    print(
                        "  Verifique se o modelo existe ou use 'modelos listar' para ver os disponíveis."
                    )

                # Ativar streaming se solicitado
                if cli and args.stream:
                    cli.streaming_enabled = True
                    print("Streaming de respostas ativado")
            except Exception as e:
                logger.error(f"Erro durante inicialização do modelo: {str(e)}")
                print(f"⚠ Erro durante inicialização: {str(e)}")
                print(
                    "  O Èter continuará funcionando, mas sem recursos avançados de IA."
                )
                if cli:
                    cli.streaming_enabled = False

        # Iniciar loop de comando apenas se o CLI foi inicializado corretamente
        if cli:
            cli.cmdloop()
        else:
            print("Erro: CLI não pôde ser inicializado corretamente.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nÈter encerrado pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"Erro ao inicializar Èter: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
