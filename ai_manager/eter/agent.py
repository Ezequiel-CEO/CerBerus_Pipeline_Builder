#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo principal do agente Èter para o CerBerus FMK

Este módulo implementa a classe principal EterAgent, que coordena todas as
funcionalidades do assistente inteligente, incluindo gerenciamento de memória,
aprendizado, processamento de linguagem natural e execução de tarefas.
"""

import os
import sys
import platform
import json
import logging
import datetime
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Generator

from cerberus_api.pipeline_builder.ai_manager.eter.context.memory import EterMemory
from cerberus_api.utils.logging_config import get_logger

# Importar ferramentas
try:
    from cerberus_api.pipeline_builder.ai_manager.eter.tools.shell import ShellCommands
    from cerberus_api.pipeline_builder.ai_manager.eter.tools.files import FileCommands

    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False

# Verificar se temos suporte a modelos GGUF
GGUF_AVAILABLE = False
try:
    from cerberus_api.pipeline_builder.ai_manager.llm.gguf_manager import GGUFManager

    GGUF_AVAILABLE = True
except ImportError:
    pass

# Importar nossa ponte do framework
try:
    from cerberus_api.pipeline_builder.ai_manager.eter.core_integration import (
        eter_bridge,
    )

    CORE_INTEGRATION_AVAILABLE = True
except ImportError:
    CORE_INTEGRATION_AVAILABLE = False


class EterAgent:
    """
    Agente inteligente do Èter, responsável por gerenciar os componentes
    do assistente, processar entradas do usuário e coordenar a execução
    de tarefas.

    Responsável pelo gerenciamento de:
    - Memória e contexto
    - Aprendizado e adaptação
    - Processamento de linguagem natural
    - Execução de tarefas
    """

    def __init__(
        self,
        user_id: str = "default",
        model_dir: str = None,
        knowledge_dir: str = None,
        workspace_dir: str = None,
        verbose: bool = False,
        enable_framework_bridge: bool = True,
        **kwargs,
    ):
        """
        Inicializa o agente Èter.

        Args:
            user_id: Identificador do usuário
            model_dir: Diretório de modelos
            knowledge_dir: Diretório de base de conhecimento
            workspace_dir: Diretório de trabalho
            verbose: Ativar modo verboso
            enable_framework_bridge: Ativar integração com framework
            **kwargs: Parâmetros adicionais
        """
        self.user_id = user_id
        self.verbose = verbose

        # Configurar diretórios
        self.model_dir = model_dir or os.environ.get(
            "ETER_MODEL_DIR", os.path.expanduser("~/models")
        )
        self.knowledge_dir = knowledge_dir or os.environ.get(
            "ETER_KNOWLEDGE_DIR", os.path.expanduser("~/knowledge")
        )
        self.workspace_dir = workspace_dir or os.environ.get(
            "ETER_WORKSPACE_DIR", os.getcwd()
        )

        # Configurar logging
        self.logger = logging.getLogger(f"eter.agent.{user_id}")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            log_level = kwargs.get("log_level", "INFO")
            self.logger.setLevel(getattr(logging, log_level.upper()))

        self.logger.info(f"Inicializando agente Èter para usuário: {user_id}")
        self.logger.info(
            f"Diretórios: modelo={self.model_dir}, conhecimento={self.knowledge_dir}, workspace={self.workspace_dir}"
        )

        # Verificar disponibilidade de recursos
        self.gguf_available = False
        try:
            # Tentar importar o módulo gguf_loader - se não existir, vai falhar graciosamente
            from cerberus_api.pipeline_builder.ai_manager.eter.gguf_loader import (
                GGUFModel,
            )

            self.gguf_available = True
            self.logger.info("Suporte a modelos GGUF disponível")
        except ImportError:
            self.logger.warning("Suporte a modelos GGUF não disponível")

        # Tentar importar a ponte do framework
        CORE_INTEGRATION_AVAILABLE = False
        try:
            from cerberus_api.pipeline_builder.ai_manager.eter.core_integration import (
                eter_bridge,
            )

            CORE_INTEGRATION_AVAILABLE = True
            self.logger.info("Integração com o framework disponível")
        except ImportError:
            self.logger.warning("Módulo de integração com o framework não encontrado")
        except Exception as e:
            self.logger.error(f"Erro ao importar ponte do framework: {e}")

        # Configurar ponte do framework
        self.has_framework_bridge = (
            CORE_INTEGRATION_AVAILABLE and enable_framework_bridge
        )
        if self.has_framework_bridge:
            from cerberus_api.pipeline_builder.ai_manager.eter.core_integration import (
                eter_bridge,
            )

            self.framework = eter_bridge
            self.logger.info(f"Ponte do framework inicializada para usuário: {user_id}")
        else:
            self.framework = None
            if enable_framework_bridge:
                self.logger.warning("Ponte do framework solicitada mas não disponível")
            else:
                self.logger.info("Ponte do framework desativada conforme configuração")

        # Verificar dependências opcionais
        self.has_llm_support = False

        # Inicializar gerenciador de modelos LLM (se disponível)
        try:
            from cerberus_api.pipeline_builder.ai_manager.model_hub import ModelHub

            self.llm_manager = ModelHub(models_dir=self.model_dir)
            self.has_llm_support = True
        except ImportError:
            self.logger.warning(
                "Suporte a LLM não disponível. Alguns recursos serão limitados."
            )
            self.llm_manager = None

        # Componentes principais (inicializados sob demanda)
        self.memory = None
        self.active_model = None

        # Inicializar componentes essenciais
        self._initialize_components()

        # Ferramentas e utilitários
        self._initialize_tools()

    def _initialize_components(self):
        """Inicializa os componentes principais do agente."""
        try:
            # Inicializar memória
            from cerberus_api.pipeline_builder.ai_manager.eter.context.memory import (
                EterMemory,
            )

            self.memory = EterMemory(
                storage_dir=self.knowledge_dir, user_id=self.user_id
            )
            self.logger.info("Memória inicializada")

            # Verificar se temos o ModelHub disponível
            if hasattr(self, "llm_manager") and self.llm_manager:

                # Tentar carregar modelos GGUF padrão se disponíveis
                try:
                    # Verificar se temos modelos GGUF disponíveis
                    default_models_dir = os.path.expanduser("~/.cerberusfmk/models/")

                    self.logger.info(f"Verificando modelos em: {default_models_dir}")

                    # Adicionar modelo Qwen2.5 se disponível
                    model_found = False
                    for model_name in [
                        "Qwen2.5-7B-Instruct-1M-Q6_K.gguf",
                        "Qwen2.5-7B-Instruct-1M-Q6_K.final.gguf",
                    ]:
                        qwen_model_path = os.path.join(default_models_dir, model_name)
                        if os.path.exists(qwen_model_path):
                            try:
                                # Extrair nome do modelo do arquivo
                                model_name_clean = "Qwen2.5-7B-Instruct-1M-Q6_K"
                                # Adicionar com nome e caminho separados
                                self.llm_manager.add_model(qwen_model_path)
                                self.logger.info(
                                    f"Modelo registrado: {model_name_clean}"
                                )
                                # Definir como modelo ativo automaticamente
                                self.active_model_name = model_name_clean
                                model_found = True
                                break
                            except Exception as e:
                                self.logger.warning(
                                    f"Erro ao registrar modelo '{model_name_clean}': {e}"
                                )

                    # Se encontramos o modelo, carregar
                    if model_found:
                        self.logger.info("Tentando definir o modelo como ativo...")
                        try:
                            self.set_active_model(
                                "Qwen2.5-7B-Instruct-1M-Q6_K", force_reload=True
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Erro ao definir modelo como ativo: {e}"
                            )
                except Exception as e:
                    self.logger.warning(f"Erro ao verificar modelos padrão: {e}")

            # Inicializar módulos de aceleração
            try:
                import cerberus_api.pipeline_builder.ai_manager.raw_gpu

                self.logger.info("Módulo raw_gpu disponível para aceleração")
            except ImportError:
                self.logger.warning("Módulo raw_gpu não disponível")

        except Exception as e:
            self.logger.error(f"Erro ao inicializar componentes: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

    def _initialize_tools(self):
        """Inicializa as ferramentas e utilitários do agente."""
        # Importar ferramentas
        try:
            from cerberus_api.pipeline_builder.ai_manager.eter.tools.shell import (
                ShellTools,
            )
            from cerberus_api.pipeline_builder.ai_manager.eter.tools.files import (
                FileTools,
            )

            # Inicializar ferramentas
            self.shell_tools = ShellTools(self)
            self.file_tools = FileTools(self)

            self.logger.info("Ferramentas de shell e arquivos disponíveis")
        except ImportError:
            self.logger.warning("Ferramentas opcionais não disponíveis")
            self.shell_tools = None
            self.file_tools = None

    def set_active_model(self, model_name, force_reload=False):
        """
        Define o modelo ativo para geração de texto.

        Args:
            model_name: Nome do modelo ou caminho completo para o arquivo .gguf
            force_reload: Se True, recarrega o modelo mesmo se já estiver carregado

        Returns:
            True se o modelo foi ativado com sucesso
        """
        self.logger.info(f"Configurando modelo ativo: {model_name}")

        # Verificar se o modelo_name é um caminho completo para um arquivo
        if os.path.isfile(model_name) and model_name.endswith(".gguf"):
            model_path = model_name
            # Extrair nome do modelo do caminho
            base_name = os.path.basename(model_path)
            model_name = os.path.splitext(base_name)[0]
            self.logger.info(
                f"Caminho direto para modelo detectado. Nome: {model_name}, Caminho: {model_path}"
            )
        else:
            # Verificar o diretório de modelos personalizado
            custom_models_dir = os.path.expanduser("~/.cerberusfmk/models/")

            # Construir caminhos possíveis para o modelo
            possible_paths = [
                os.path.join(custom_models_dir, f"{model_name}.gguf"),
                os.path.join(custom_models_dir, model_name),
                os.path.join(self.model_dir, f"{model_name}.gguf"),
                os.path.join(self.model_dir, model_name),
            ]

            # Verificar qual caminho existe
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

        if model_path is None:
            self.logger.warning(f"Arquivo de modelo não encontrado para: {model_name}")
            return False

        try:
            # Verificar se temos um gerenciador de modelos
            if not self.llm_manager:
                self.logger.warning("Gerenciador de modelos não disponível")
                return False

            # Verificar se o modelo já está carregado
            if not force_reload:
                # Verificar se temos um modelo ativo e se é o mesmo que estamos tentando definir
                if (
                    hasattr(self, "active_model_name")
                    and self.active_model_name == model_name
                ):
                    self.logger.info(f"Modelo '{model_name}' já está ativo")
                    return True

            # Primeiro registrar o modelo se necessário
            # Verificar se o modelo está registrado
            try:
                if hasattr(self.llm_manager, "add_model"):
                    # Verificar quantos parâmetros aceita o método add_model
                    import inspect

                    try:
                        add_model_params = inspect.signature(
                            self.llm_manager.add_model
                        ).parameters
                        param_count = len(add_model_params)

                        if param_count >= 2:
                            # API espera nome e caminho separados
                            self.llm_manager.add_model(
                                model_name, model_path, model_type="gguf"
                            )
                        else:
                            # API espera apenas o caminho
                            self.llm_manager.add_model(model_path)
                    except Exception:
                        # Em caso de erro, tentar o método mais simples
                        self.llm_manager.add_model(model_path)

                    self.logger.info(f"Modelo registrado: {model_name}")
            except Exception as e:
                self.logger.warning(f"Erro ao registrar modelo '{model_name}': {e}")

            # Em seguida, carregá-lo
            success = False
            if hasattr(self.llm_manager, "load_model"):
                success = self.llm_manager.load_model(model_name)

                if success:
                    # Armazenar nome do modelo para futuras referências
                    self.active_model_name = model_name
                    self.logger.info(f"Modelo '{model_name}' carregado com sucesso")
                else:
                    self.logger.error(f"Falha ao carregar modelo '{model_name}'")
            else:
                self.logger.warning(
                    "Método load_model não disponível no gerenciador de modelos"
                )
                return False

            return success
        except Exception as e:
            self.logger.error(f"Erro ao definir modelo ativo: {e}")
            return False

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Lista todos os modelos disponíveis.

        Returns:
            Lista de dicionários com informações dos modelos
        """
        models = []

        try:
            # Primeiro, verificar modelos no diretório padrão
            home_models_dir = os.path.expanduser("~/.cerberusfmk/models/")
            self.logger.info(f"Verificando modelos em: {home_models_dir}")

            if not os.path.exists(home_models_dir):
                os.makedirs(home_models_dir, exist_ok=True)

            # Verificar se o ModelHub está disponível
            if self.llm_manager is not None:
                # Verificar se o método list_models existe no ModelHub
                if hasattr(self.llm_manager, "list_models"):
                    try:
                        # Usar o método list_models do ModelHub
                        models = self.llm_manager.list_models()
                        if models:
                            return models
                    except Exception as e:
                        self.logger.error(
                            f"Erro ao listar modelos via ModelHub: {str(e)}"
                        )
                        # Continuar com a implementação de fallback

            # Fallback: listar arquivos GGUF diretamente
            gguf_files = []
            for root, _, files in os.walk(home_models_dir):
                for file in files:
                    if file.lower().endswith(".gguf"):
                        model_path = os.path.join(root, file)

                        # Obter informações básicas do arquivo
                        try:
                            size = os.path.getsize(model_path)
                            name = os.path.splitext(file)[0]

                            model_info = {
                                "id": name.lower().replace(" ", "_"),
                                "name": name,
                                "path": model_path,
                                "size": size,
                                "size_gb": size / (1024**3),
                            }

                            gguf_files.append(model_info)
                        except OSError:
                            # Ignorar arquivos com problemas
                            continue

            # Ordenar por tamanho (maior primeiro)
            models = sorted(gguf_files, key=lambda x: x.get("size", 0), reverse=True)

            # Registrar modelos no ModelHub se disponível
            if self.llm_manager is not None and hasattr(self.llm_manager, "add_model"):
                for model in models:
                    if model.get("path"):
                        try:
                            self.llm_manager.add_model(model["path"])
                        except Exception as e:
                            self.logger.warning(
                                f"Erro ao registrar modelo '{model['name']}': {str(e)}"
                            )

        except Exception as e:
            self.logger.error(f"Erro ao listar modelos: {str(e)}")

        return models

    def _generate_response(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        streaming: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Gera uma resposta utilizando o modelo ativo.

        Args:
            prompt: Texto para o modelo responder
            max_tokens: Número máximo de tokens na resposta
            temperature: Temperatura para sampling (maior = mais aleatório)
            streaming: Se True, retorna um gerador com chunks da resposta

        Returns:
            Texto da resposta ou gerador de chunks
        """
        # Verificar se temos acesso a modelos LLM
        if self.active_model is None:
            # Resposta de fallback quando não há modelo disponível
            fallback = (
                "Não foi possível processar sua solicitação com um modelo avançado."
            )
            if streaming:
                return (chunk for chunk in [fallback])
            return fallback

        # Preparar o prompt com histórico de conversa
        conversation_history = self.memory.get_conversation_history(limit=5)

        # Formatar histórico para o prompt
        formatted_history = ""
        for entry in conversation_history:
            role = entry.get("role", "")
            content = entry.get("content", "")

            if role == "user":
                formatted_history += f"Usuário: {content}\n"
            elif role == "assistant":
                formatted_history += f"Èter: {content}\n"

        # Adicionar sistema de personalidade e contexto
        system_prompt = """Você é Èter, um assistente de IA avançado projetado para gerenciar o framework CerBerus.
Você deve ser direto, profissional e útil. 
Responda sempre em português brasileiro.
Você foi criado para ajudar com gerenciamento de modelos de IA, processamento de dados e análise de informações.
Use sua vasta base de conhecimento para auxiliar o usuário da melhor forma possível.
"""

        # Estruturar o prompt completo
        full_prompt = f"{system_prompt}\n\nHistórico de conversa:\n{formatted_history}\n\nUsuário: {prompt}\n\nÈter:"

        try:
            # Gerar resposta usando o modelo
            return self.active_model.generate(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                streaming=streaming,
            )
        except Exception as e:
            # Em caso de erro, fornecer uma resposta alternativa
            error_msg = (
                f"Desculpe, ocorreu um erro ao processar sua solicitação: {str(e)}"
            )
            if streaming:
                return (chunk for chunk in [error_msg])
            return error_msg

    def process_input(self, user_input, use_simulated=None):
        """
        Processa a entrada do usuário e retorna uma resposta.

        Args:
            user_input (str): A entrada do usuário para processar
            use_simulated (bool, opcional): Se deve usar modo simulado, independente do estado do modelo

        Returns:
            str: A resposta gerada
        """
        self._logger.info(f"Processando entrada: {user_input[:20]}...")

        # Decidir se usamos modo simulado
        should_simulate = (
            use_simulated if use_simulated is not None else not self.is_model_active()
        )

        if not should_simulate:
            try:
                # Usar o modelo ativo para gerar uma resposta
                self._logger.info(f"Usando modelo ativo: {self._active_model}")

                if hasattr(self._model_hub, "generate_text"):
                    response = self._model_hub.generate_text(
                        self._active_model,
                        f"Usuário: {user_input}\n\nAjudante:",
                        max_tokens=1024,
                        temperature=0.7,
                        top_p=0.95,
                    )
                    return response
                else:
                    should_simulate = True  # Fallback para simulação
                    self._logger.warning(
                        "ModelHub não tem método generate_text, usando simulação"
                    )
            except Exception as e:
                should_simulate = True  # Fallback para simulação em caso de erro
                self._logger.error(f"Erro ao gerar resposta com modelo: {e}")

        if should_simulate:
            # Modo simulado quando não há modelo disponível
            return f"[SIMULAÇÃO] Resposta para: Usuário: {user_input}\n\nAjudante:"

    def _generate_fallback_response(self, user_input: str) -> str:
        """
        Gera uma resposta de fallback quando o LLM não está disponível.

        Args:
            user_input: Texto de entrada do usuário

        Returns:
            Resposta de fallback
        """
        # Respostas simples baseadas em palavras-chave
        lower_input = user_input.lower()

        if "olá" in lower_input or "oi" in lower_input or "ola" in lower_input:
            return "Olá! Como posso ajudar você hoje?"

        if "ajuda" in lower_input:
            return (
                "Estou aqui para ajudar. O que você precisa saber sobre o CerBerus FMK?"
            )

        if "quem" in lower_input and ("você" in lower_input or "voce" in lower_input):
            return "Eu sou Èter, o assistente do CerBerus FMK, projetado para ajudar com várias tarefas de processamento de linguagem natural, análise de dados e automação."

        # Resposta genérica
        return f"Entendi sua consulta. Você está perguntando sobre '{user_input}'. Posso ajudar fornecendo mais informações se você detalhar melhor sua pergunta."

    def execute_task(self, task_description: str) -> Dict[str, Any]:
        """
        Executa uma tarefa descrita em linguagem natural.

        Args:
            task_description: Descrição da tarefa em linguagem natural

        Returns:
            Dict[str, Any]: Resultado da execução da tarefa
        """
        self.logger.info(f"Tentando executar tarefa: {task_description}")

        if not task_description or task_description.strip() == "":
            return {
                "status": "error",
                "message": "Descrição da tarefa vazia",
                "result": None,
            }

        # Registrar a tarefa na memória
        self.memory.add_conversation(
            role="user",
            content=f"Executar tarefa: {task_description}",
            context_id="tasks",
            metadata={"type": "task", "timestamp": datetime.now().isoformat()},
        )

        try:
            # 1. Analisar o tipo de tarefa
            task_type = self._classify_task(task_description)

            # 2. Executar com base no tipo de tarefa
            result = self._dispatch_task(task_type, task_description)

            # 3. Registrar resultado na memória
            self.memory.add_conversation(
                role="assistant",
                content=f"Resultado da tarefa: {result.get('message', 'Concluído')}",
                context_id="tasks",
                metadata={
                    "type": "task_result",
                    "task_type": task_type,
                    "status": result.get("status", "unknown"),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            return result
        except Exception as e:
            self.logger.error(f"Erro ao executar tarefa: {e}")
            error_result = {
                "status": "error",
                "message": f"Erro ao executar tarefa: {str(e)}",
                "result": None,
            }

            # Registrar erro na memória
            self.memory.add_conversation(
                role="assistant",
                content=f"Erro na execução da tarefa: {str(e)}",
                context_id="tasks",
                metadata={
                    "type": "task_error",
                    "timestamp": datetime.now().isoformat(),
                },
            )

            return error_result

    def _classify_task(self, task_description: str) -> str:
        """
        Classifica o tipo de tarefa com base na descrição.

        Args:
            task_description: Descrição da tarefa

        Returns:
            str: Tipo de tarefa identificado
        """
        task_description = task_description.lower()

        # Mapeamento de palavras-chave para tipos de tarefa
        task_keywords = {
            "arquivo": "file_operation",
            "diretório": "file_operation",
            "ler": "file_operation",
            "escrever": "file_operation",
            "abrir": "file_operation",
            "salvar": "file_operation",
            "executar": "command_execution",
            "comando": "command_execution",
            "rodar": "command_execution",
            "terminal": "command_execution",
            "pipeline": "pipeline_management",
            "modelo": "model_management",
            "treinar": "model_training",
            "treinamento": "model_training",
            "coletar": "data_collection",
            "dados": "data_operation",
            "análise": "data_analysis",
            "analisar": "data_analysis",
            "status": "system_status",
            "estatística": "system_status",
            "uso": "system_status",
        }

        # Verificar ocorrências de palavras-chave
        task_scores = {}
        for keyword, task_type in task_keywords.items():
            if keyword in task_description:
                task_scores[task_type] = task_scores.get(task_type, 0) + 1

        # Selecionar o tipo com maior pontuação
        if task_scores:
            return max(task_scores.items(), key=lambda x: x[1])[0]

        # Tipo padrão se nenhuma palavra-chave for encontrada
        return "general_task"

    def _dispatch_task(self, task_type: str, task_description: str) -> Dict[str, Any]:
        """
        Distribui a execução da tarefa com base no tipo identificado.

        Args:
            task_type: Tipo da tarefa
            task_description: Descrição da tarefa

        Returns:
            Dict[str, Any]: Resultado da execução
        """
        self.logger.info(f"Distribuindo tarefa do tipo: {task_type}")

        # Verificar se temos acesso ao framework
        if self.framework_bridge and hasattr(self.framework_bridge, "get_component"):
            # Executar tarefas específicas do framework
            if task_type == "pipeline_management":
                return self._execute_pipeline_task(task_description)

            elif task_type == "model_management" or task_type == "model_training":
                return self._execute_model_task(task_description)

            elif task_type == "data_collection" or task_type == "data_analysis":
                return self._execute_data_task(task_description)

        # Tarefas básicas que não precisam do framework
        if task_type == "file_operation":
            return self._execute_file_task(task_description)

        elif task_type == "command_execution":
            return self._execute_command_task(task_description)

        elif task_type == "system_status":
            return self._get_system_status()

        # Tarefa não implementada ou não reconhecida
        return {
            "status": "not_implemented",
            "message": "A execução desta tarefa ainda não está implementada",
            "task_type": task_type,
            "task_description": task_description,
        }

    def _execute_file_task(self, task_description: str) -> Dict[str, Any]:
        """Executa operações com arquivos"""
        # Implementação básica para listar arquivos do diretório atual
        if "listar" in task_description and (
            "arquivo" in task_description or "diretório" in task_description
        ):
            dir_path = self.workspace_dir
            try:
                files = os.listdir(dir_path)
                return {
                    "status": "success",
                    "message": f"Arquivos listados com sucesso em: {dir_path}",
                    "result": {
                        "directory": dir_path,
                        "files": files,
                        "count": len(files),
                    },
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Erro ao listar arquivos: {str(e)}",
                    "result": None,
                }

        return {
            "status": "not_implemented",
            "message": "Este tipo específico de operação com arquivos ainda não está implementado",
            "result": None,
        }

    def _execute_command_task(self, task_description: str) -> Dict[str, Any]:
        """Executa comandos de terminal"""
        # Esta é uma implementação simplificada e limitada
        # Em um sistema real, seria necessário extrair o comando da descrição
        # Para agora, apenas indicamos que não está completamente implementado
        return {
            "status": "not_implemented",
            "message": "A execução de comandos a partir de descrições ainda não está totalmente implementada",
            "result": None,
        }

    def _execute_pipeline_task(self, task_description: str) -> Dict[str, Any]:
        """Executa tarefas relacionadas a pipelines"""
        try:
            if "listar" in task_description:
                # Tentar obter o PipelineManager
                pipeline_manager = self.framework_bridge.get_component(
                    "cerberus_api.pipeline_builder.core.PipelineManager"
                )
                if pipeline_manager:
                    pipelines = pipeline_manager.list_pipelines()
                    return {
                        "status": "success",
                        "message": f"Pipelines listados com sucesso: {len(pipelines)} encontrados",
                        "result": {"pipelines": pipelines, "count": len(pipelines)},
                    }

            return {
                "status": "not_implemented",
                "message": "Esta operação específica de pipeline ainda não está implementada",
                "result": None,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erro ao executar operação de pipeline: {str(e)}",
                "result": None,
            }

    def _get_system_status(self) -> Dict[str, Any]:
        """Obtém informações de status do sistema"""
        try:
            import psutil

            # Informações básicas do sistema
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage("/")

            # Verificar disponibilidade de GPU
            gpu_info = {"available": False}
            try:
                import GPUtil

                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Usar primeira GPU
                    gpu_info = {
                        "available": True,
                        "name": gpu.name,
                        "memory_total": f"{gpu.memoryTotal:.2f} MB",
                        "memory_used": f"{gpu.memoryUsed:.2f} MB",
                        "memory_free": f"{gpu.memoryFree:.2f} MB",
                        "memory_percent": f"{gpu.memoryUtil * 100:.1f}%",
                        "temperature": f"{gpu.temperature}°C",
                        "load": f"{gpu.load * 100:.1f}%",
                    }
            except:
                pass

            return {
                "status": "success",
                "message": "Informações do sistema obtidas com sucesso",
                "result": {
                    "cpu": {
                        "percent": cpu_percent,
                        "cores": psutil.cpu_count(logical=True),
                    },
                    "memory": {
                        "total_gb": memory.total / (1024**3),
                        "available_gb": memory.available / (1024**3),
                        "percent": memory.percent,
                    },
                    "disk": {
                        "total_gb": disk.total / (1024**3),
                        "free_gb": disk.free / (1024**3),
                        "percent": disk.percent,
                    },
                    "gpu": gpu_info,
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                },
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erro ao obter informações do sistema: {str(e)}",
                "result": None,
            }

    def _execute_model_task(self, task_description: str) -> Dict[str, Any]:
        """Executa tarefas relacionadas a modelos"""
        try:
            if "listar" in task_description and "modelo" in task_description:
                # Listar modelos disponíveis
                models = self.list_available_models()
                return {
                    "status": "success",
                    "message": f"Modelos listados com sucesso: {len(models)} encontrados",
                    "result": {"models": models, "count": len(models)},
                }

            return {
                "status": "not_implemented",
                "message": "Esta operação específica de modelo ainda não está implementada",
                "result": None,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erro ao executar operação de modelo: {str(e)}",
                "result": None,
            }

    def _execute_data_task(self, task_description: str) -> Dict[str, Any]:
        """Executa tarefas relacionadas a dados"""
        # Esta é uma função placeholder para operações de dados
        return {
            "status": "not_implemented",
            "message": "A execução de tarefas de dados ainda não está implementada",
            "result": None,
        }

    def is_model_active(self) -> bool:
        """
        Verifica se há um modelo ativo e carregado.

        Returns:
            bool: True se há um modelo ativo e carregado, False caso contrário.
        """
        if not self.llm_manager:
            return False

        # Verificar se há um modelo ativo
        if (
            not hasattr(self.llm_manager, "active_model")
            or not self.llm_manager.active_model
        ):
            return False

        # Verificar se o modelo está carregado
        if hasattr(self.llm_manager, "is_model_loaded"):
            return self.llm_manager.is_model_loaded(self.llm_manager.active_model)

        # Se não conseguir verificar diretamente, assumir que se tem um active_model, está carregado
        return True

    def __del__(self):
        """
        Método destrutor com limpeza segura de recursos CUDA.
        """
        try:
            # Liberar explicitamente recursos do modelo ativo
            if hasattr(self, "_active_model") and self._active_model:
                try:
                    if hasattr(self._model_hub, "unload_model"):
                        self._model_hub.unload_model(self._active_model)
                    self._active_model = None
                except Exception as e:
                    print(f"Aviso: Erro ao descarregar modelo: {e}")

            # Limpar recursos CUDA explicitamente
            try:
                import gc

                gc.collect()

                # Descarregar módulos CUDA
                for name in list(sys.modules.keys()):
                    if (
                        "cuda" in name.lower()
                        or "gpu" in name.lower()
                        or "llama" in name.lower()
                    ):
                        if name in sys.modules:
                            try:
                                del sys.modules[name]
                            except:
                                pass
            except Exception as e:
                print(f"Aviso: Erro na limpeza de módulos CUDA: {e}")

        except Exception as e:
            print(f"Aviso: Erro no destrutor do EterAgent: {e}")
