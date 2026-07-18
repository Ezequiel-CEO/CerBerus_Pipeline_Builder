#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Èter Core Integration: Integração central do Èter com o CerBerusFMK.

Este módulo implementa a ponte entre o agente Èter e todos os componentes
do framework CerBerusFMK, permitindo que o Èter opere como o núcleo central
do sistema, acessando, controlando e estendendo todas as funcionalidades.
"""

import os
import sys
import time
import logging
import importlib
import inspect
from typing import Dict, List, Any, Optional, Union, Callable, Type, Tuple
from pathlib import Path
import uuid

from cerberus_api.utils.logging_config import get_logger

# Importa o módulo de aceleração CUDA
from cerberus_api.pipeline_builder.ai_manager.eter.cuda_accelerator import (
    integrate_cuda_accelerator,
)

# Logger configurado
logger = get_logger("eter_core_integration")


class EterComponentRegistry:
    """
    Registro de componentes do framework para acesso pelo Èter.

    Esta classe mapeia e indexa todos os módulos, classes e funções
    disponíveis no CerBerusFMK para permitir que o Èter os acesse
    dinamicamente sem depender de imports hardcoded.
    """

    def __init__(self):
        """Inicializa o registro de componentes."""
        self.modules = {}
        self.classes = {}
        self.functions = {}
        self.instances = {}
        self.apis = {}

        # Raiz do framework para busca relativa
        self.framework_root = self._find_framework_root()

        logger.info(
            f"Registro de componentes inicializado. Root: {self.framework_root}"
        )

    def _find_framework_root(self) -> str:
        """
        Encontra o diretório raiz do framework CerBerusFMK.

        Returns:
            Caminho para o diretório raiz do framework
        """
        # Tentar encontrar pela estrutura de importação
        try:
            import cerberus_api

            return os.path.dirname(os.path.dirname(cerberus_api.__file__))
        except ImportError:
            pass

        # Alternativa: buscar pelo diretório atual
        current_dir = os.path.abspath(os.path.dirname(__file__))

        # Procurar por "CerBerusFMK" no caminho
        parts = current_dir.split(os.sep)
        for i in range(len(parts), 0, -1):
            path = os.sep.join(parts[:i])
            if os.path.basename(path) == "CerBerusFMK":
                return path

        # Se não encontrar, usar o diretório atual
        return os.path.dirname(current_dir)

    def discover_components(self, scan_paths: Optional[List[str]] = None) -> None:
        """
        Descobre automaticamente os componentes disponíveis no framework.

        Args:
            scan_paths: Lista de caminhos a serem escaneados (relativo à raiz)
        """
        logger.info("Iniciando descoberta de componentes do framework...")

        # Caminhos padrão para escaneamento
        if scan_paths is None:
            scan_paths = [
                "cerberus_api/pipeline_builder",
                "cerberus_api/utils",
                "cerberus_api/models",
            ]

        # Converter para caminhos absolutos
        abs_paths = [os.path.join(self.framework_root, path) for path in scan_paths]

        # Escanear cada caminho
        for path in abs_paths:
            if os.path.exists(path):
                self._scan_directory(path)
            else:
                logger.warning(f"Caminho não encontrado: {path}")

        logger.info(
            f"Descoberta concluída: {len(self.modules)} módulos, {len(self.classes)} classes, {len(self.functions)} funções"
        )

    def _scan_directory(self, directory: str) -> None:
        """
        Escaneia um diretório em busca de módulos Python.

        Args:
            directory: Caminho do diretório a ser escaneado
        """
        if not os.path.exists(directory):
            return

        for item in os.listdir(directory):
            path = os.path.join(directory, item)

            # Ignorar diretórios ocultos e __pycache__
            if item.startswith(".") or item == "__pycache__":
                continue

            # Recursão para subdiretórios
            if os.path.isdir(path):
                # Verificar se é um pacote Python
                if os.path.exists(os.path.join(path, "__init__.py")):
                    self._scan_directory(path)

            # Importar módulos Python
            elif item.endswith(".py"):
                # Transformar caminho de arquivo em module path
                rel_path = os.path.relpath(path, self.framework_root)
                module_path = rel_path.replace(os.path.sep, ".").replace(".py", "")

                # Tentar importar o módulo
                self._import_module(module_path)

    def _import_module(self, module_path: str) -> None:
        """
        Importa um módulo e registra seus componentes.

        Args:
            module_path: Caminho de importação do módulo
        """
        try:
            module = importlib.import_module(module_path)

            # Registrar o módulo
            self.modules[module_path] = module

            # Escanear classes e funções
            for name, obj in inspect.getmembers(module):
                # Ignorar objetos privados e importados de outros módulos
                if name.startswith("_"):
                    continue

                if inspect.isclass(obj):
                    # Verificar se a classe é definida neste módulo
                    if obj.__module__ == module_path:
                        self.classes[f"{module_path}.{name}"] = obj

                elif inspect.isfunction(obj):
                    # Verificar se a função é definida neste módulo
                    if obj.__module__ == module_path:
                        self.functions[f"{module_path}.{name}"] = obj

            logger.debug(f"Módulo importado: {module_path}")

        except Exception as e:
            logger.debug(f"Erro ao importar módulo {module_path}: {e}")

    def get_component(self, component_path: str) -> Any:
        """
        Obtém um componente pelo seu caminho.

        Args:
            component_path: Caminho para o componente (módulo, classe ou função)

        Returns:
            Componente solicitado ou None se não encontrado
        """
        # Verificar cache de instâncias
        if component_path in self.instances:
            return self.instances[component_path]

        # Verificar classes
        if component_path in self.classes:
            return self.classes[component_path]

        # Verificar funções
        if component_path in self.functions:
            return self.functions[component_path]

        # Verificar módulos
        if component_path in self.modules:
            return self.modules[component_path]

        # Tentar importar diretamente
        try:
            parts = component_path.split(".")
            for i in range(len(parts), 0, -1):
                # Tentar importar o módulo mais externo possível
                module_path = ".".join(parts[:i])
                try:
                    module = importlib.import_module(module_path)
                    remaining = ".".join(parts[i:])

                    # Se não há partes restantes, retornar o módulo
                    if not remaining:
                        return module

                    # Caso contrário, navegar no módulo
                    obj = module
                    for part in parts[i:]:
                        obj = getattr(obj, part)
                    return obj

                except (ImportError, AttributeError):
                    continue

        except Exception as e:
            logger.debug(f"Erro ao obter componente {component_path}: {e}")

        return None

    def create_instance(self, class_path: str, *args, **kwargs) -> Any:
        """
        Cria uma instância de uma classe pelo seu caminho.

        Args:
            class_path: Caminho para a classe
            *args: Argumentos posicionais para o construtor
            **kwargs: Argumentos nomeados para o construtor

        Returns:
            Instância da classe ou None se falhar
        """
        # Obter a classe
        cls = self.get_component(class_path)
        if cls is None or not inspect.isclass(cls):
            logger.error(f"Classe não encontrada: {class_path}")
            return None

        # Criar instância
        try:
            instance = cls(*args, **kwargs)
            instance_id = f"{class_path}@{id(instance)}"
            self.instances[instance_id] = instance
            return instance
        except Exception as e:
            logger.error(f"Erro ao instanciar {class_path}: {e}")
            return None

    def cache_instance(self, instance: Any, name: Optional[str] = None) -> str:
        """
        Armazena uma instância no cache para uso futuro.

        Args:
            instance: Instância a ser armazenada
            name: Nome opcional para a instância

        Returns:
            Identificador da instância no cache
        """
        if name is None:
            cls_name = instance.__class__.__name__
            instance_id = f"{cls_name}@{id(instance)}"
        else:
            instance_id = name

        self.instances[instance_id] = instance
        return instance_id

    def register_api(self, name: str, api_mapping: Dict[str, Callable]) -> None:
        """
        Registra uma API para acesso simplificado pelo Èter.

        Args:
            name: Nome da API
            api_mapping: Mapeamento de métodos da API
        """
        self.apis[name] = api_mapping
        logger.info(f"API registrada: {name} com {len(api_mapping)} métodos")


class EterFrameworkBridge:
    """
    Ponte de alto nível para Èter interagir com o Framework CerBerusFMK.

    Esta classe fornece uma interface simplificada para que o agente Èter possa
    acessar e controlar todos os componentes do framework.
    """

    def __init__(self):
        """Inicializa a ponte do framework."""
        self.logger = logging.getLogger("eter.framework_bridge")
        self.registry = EterComponentRegistry()
        self._registry_initialized = False
        self.logger.info("Ponte do Framework CerBerusFMK inicializada")

        # Cache de componentes frequentemente usados
        self._pipeline_builder = None
        self._model_hub = None
        self._memory_manager = None

        # Iniciar descoberta de componentes em segundo plano
        self._initialize_registry()

    def _initialize_registry(self):
        """Inicializa o registro de componentes."""
        try:
            if not self._registry_initialized:
                self.logger.info("Iniciando descoberta de componentes do framework...")
                self.registry.discover_components()
                self._registry_initialized = True

                # Obter contagens de componentes
                module_count = len(self.registry.modules)
                class_count = len(self.registry.classes)
                function_count = len(self.registry.functions)
                total_count = module_count + class_count + function_count

                self.logger.info(
                    f"Registro de componentes concluído. {total_count} componentes disponíveis "
                    f"({module_count} módulos, {class_count} classes, {function_count} funções)."
                )
        except Exception as e:
            self.logger.error(f"Erro ao inicializar registro de componentes: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

    def get_pipeline_builder(self) -> Any:
        """
        Obtém a instância do PipelineBuilder.

        Returns:
            Instância do PipelineBuilder ou None se não disponível
        """
        # Garantir que o registro está inicializado
        if not self._registry_initialized:
            self._initialize_registry()

        if self._pipeline_builder is None:
            try:
                pipeline_builder_class = self.registry.get_component(
                    "cerberus_api.pipeline_builder.core.PipelineBuilder"
                )
                if pipeline_builder_class:
                    self._pipeline_builder = pipeline_builder_class()
            except Exception as e:
                self.logger.error(f"Erro ao obter PipelineBuilder: {e}")

        return self._pipeline_builder

    def get_model_hub(self) -> Any:
        """
        Obtém a instância do ModelHub.

        Returns:
            Instância do ModelHub ou None se não disponível
        """
        # Garantir que o registro está inicializado
        if not self._registry_initialized:
            self._initialize_registry()

        if self._model_hub is None:
            try:
                model_hub_class = self.registry.get_component(
                    "cerberus_api.pipeline_builder.ai_manager.model_hub.ModelHub"
                )
                if model_hub_class:
                    self._model_hub = model_hub_class()
            except Exception as e:
                self.logger.error(f"Erro ao obter ModelHub: {e}")

        return self._model_hub

    def get_memory_manager(self) -> Any:
        """
        Obtém a instância do MemoryManager.

        Returns:
            Instância do MemoryManager ou None se não disponível
        """
        # Garantir que o registro está inicializado
        if not self._registry_initialized:
            self._initialize_registry()

        if self._memory_manager is None:
            try:
                memory_manager_class = self.registry.get_component(
                    "cerberus_api.pipeline_builder.utils.memory_manager.MemoryManager"
                )
                if memory_manager_class:
                    self._memory_manager = memory_manager_class()
            except Exception as e:
                self.logger.error(f"Erro ao obter MemoryManager: {e}")

        return self._memory_manager

    def create_pipeline(
        self, sources: List[Dict[str, Any]], name: str, description: str = ""
    ) -> Any:
        """
        Cria um pipeline de processamento.

        Args:
            sources: Lista de fontes de dados
            name: Nome do pipeline
            description: Descrição do pipeline

        Returns:
            Objeto Pipeline criado ou None se falhar
        """
        builder = self.get_pipeline_builder()
        if not builder:
            logger.error("PipelineBuilder não disponível")
            return None

        try:
            # Converter para objetos DataSource
            DataSource = self.registry.get_component(
                "cerberus_api.pipeline_builder.models.DataSource"
            )
            SourceType = self.registry.get_component(
                "cerberus_api.pipeline_builder.models.SourceType"
            )

            data_sources = []
            for source in sources:
                source_type = getattr(SourceType, source.get("source_type", "WEB"))
                data_source = DataSource(
                    name=source["name"],
                    source_type=source_type,
                    location=source["location"],
                    description=source.get("description", ""),
                )
                data_sources.append(data_source)

            # Criar pipeline
            pipeline = builder.build_pipeline(
                sources=data_sources, name=name, description=description
            )

            logger.info(f"Pipeline '{name}' criado com sucesso")
            return pipeline

        except Exception as e:
            logger.error(f"Erro ao criar pipeline: {e}")
            return None

    def analyze_content(
        self, content: Union[str, List[str]], streaming: bool = False
    ) -> Any:
        """
        Analisa conteúdo de texto usando o EnhancedAnalyzer.

        Args:
            content: String ou lista de strings para analisar
            streaming: Se deve usar processamento em streaming

        Returns:
            Resultados da análise ou None se falhar
        """
        try:
            # Obter o analisador
            EnhancedAnalyzer = self.registry.get_component(
                "cerberus_api.pipeline_builder.analyzers.enhanced_analyzer.EnhancedAnalyzer"
            )
            ContentChunk = self.registry.get_component(
                "cerberus_api.pipeline_builder.models.ContentChunk"
            )
            ContentType = self.registry.get_component(
                "cerberus_api.pipeline_builder.models.ContentType"
            )

            if not EnhancedAnalyzer or not ContentChunk or not ContentType:
                logger.error("Componentes necessários para análise não encontrados")
                return None

            # Obter o gerenciador de memória
            memory_manager = self.get_memory_manager()

            # Instanciar o analisador com opções de streaming
            analyzer = EnhancedAnalyzer(
                stream_processing=streaming, memory_efficient=True, use_gpu=True
            )

            # Preparar chunks
            chunks = []
            if isinstance(content, str):
                # Criar um único chunk com todos os campos obrigatórios
                chunk = ContentChunk(
                    task_id="eter_task_" + str(uuid.uuid4()),
                    text=content,
                    content_type=ContentType.TEXT,
                    source_location="eter_input",
                    length=len(content),
                    metadata={"source": "eter_input"},
                )
                chunks = [chunk]
            else:
                # Criar múltiplos chunks com todos os campos obrigatórios
                chunks = [
                    ContentChunk(
                        task_id="eter_task_" + str(uuid.uuid4()),
                        text=c,
                        content_type=ContentType.TEXT,
                        source_location="eter_input",
                        length=len(c),
                        metadata={"source": "eter_input"},
                    )
                    for c in content
                ]

            # Executar análise
            results = analyzer.analyze(chunks)

            return results

        except Exception as e:
            logger.error(f"Erro na análise de conteúdo: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def load_model(self, model_name: str) -> bool:
        """
        Carrega um modelo via ModelHub.

        Args:
            model_name: Nome do modelo

        Returns:
            True se o modelo foi carregado com sucesso
        """
        model_hub = self.get_model_hub()
        if not model_hub:
            logger.error("ModelHub não disponível")
            return False

        try:
            success = model_hub.load_model(model_name)
            if success:
                logger.info(f"Modelo '{model_name}' carregado com sucesso")
            else:
                logger.error(f"Falha ao carregar modelo '{model_name}'")
            return success
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False

    def execute_command(self, command: str) -> Dict[str, Any]:
        """
        Executa um comando shell e retorna o resultado.

        Args:
            command: Comando shell a ser executado

        Returns:
            Dicionário com o resultado, incluindo status, stdout e stderr
        """
        import subprocess
        import tempfile
        import time
        import os

        self.logger.info(f"Executando comando: {command}")

        # Validar comando básico (rejeitar comandos perigosos)
        dangerous_commands = ["rm -rf /", "mkfs", "dd if=/dev/zero"]
        if any(dc in command for dc in dangerous_commands):
            return {
                "status": "error",
                "message": "Comando potencialmente perigoso rejeitado por segurança",
                "stdout": "",
                "stderr": "Comando rejeitado",
                "exit_code": -1,
            }

        # Configurar log temporário para capturar saída
        log_file = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
        log_path = log_file.name
        log_file.close()

        start_time = time.time()
        result = {
            "status": "unknown",
            "message": "",
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "duration": 0,
        }

        try:
            # Executar o comando e redirecionar saída para o arquivo de log
            process = subprocess.Popen(
                f"{command} > {log_path} 2>&1", shell=True, executable="/bin/bash"
            )

            # Aguardar conclusão (com timeout)
            timeout = 60  # 60 segundos
            try:
                exit_code = process.wait(timeout=timeout)

                # Ler saída do arquivo
                with open(log_path, "r") as f:
                    output = f.read()

                duration = time.time() - start_time

                # Preparar resultado
                result = {
                    "status": "success" if exit_code == 0 else "error",
                    "message": (
                        f"Comando concluído com código {exit_code}"
                        if exit_code == 0
                        else f"Comando falhou com código {exit_code}"
                    ),
                    "stdout": output,
                    "stderr": "",  # Já incluído no stdout devido ao redirecionamento
                    "exit_code": exit_code,
                    "duration": round(duration, 2),
                }

            except subprocess.TimeoutExpired:
                process.kill()
                result = {
                    "status": "error",
                    "message": f"Comando excedeu o tempo limite de {timeout} segundos",
                    "stdout": "",
                    "stderr": "Timeout",
                    "exit_code": -1,
                    "duration": timeout,
                }

        except Exception as e:
            result = {
                "status": "error",
                "message": f"Erro ao executar comando: {str(e)}",
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "duration": time.time() - start_time,
            }

        finally:
            # Limpar arquivo temporário
            try:
                if os.path.exists(log_path):
                    os.unlink(log_path)
            except:
                pass

        # Registrar resultado
        log_level = logging.INFO if result["status"] == "success" else logging.ERROR
        self.logger.log(log_level, f"Comando concluído: {result['message']}")

        return result

    def list_available_components(self) -> Dict[str, List[str]]:
        """
        Lista todos os componentes disponíveis no framework.

        Returns:
            Dicionário com listas de módulos, classes e funções
        """
        return {
            "modules": list(self.registry.modules.keys()),
            "classes": list(self.registry.classes.keys()),
            "functions": list(self.registry.functions.keys()),
            "instances": list(self.registry.instances.keys()),
            "apis": list(self.registry.apis.keys()),
        }


class EterIntegration:
    def __init__(self, engine=None):
        """Inicializa integração do Éter com o CerBerus Engine"""
        self.engine = engine
        self.logger = engine.logger if engine else None
        self.cuda_accelerator = None

        # Inicializa componentes do Éter
        self.logger.info("Éter: Iniciando integração com CerBerus Engine")

        # Ativa o acelerador CUDA nativo se possível
        self._setup_cuda_accelerator()

        # Setup de outras integrações...
        # ... existing code ...

    def _setup_cuda_accelerator(self):
        """Configura acelerador CUDA para integração nativa"""
        if self.engine is None:
            return

        try:
            self.logger.info("Éter: Ativando acelerador CUDA nativo")
            self.cuda_accelerator = integrate_cuda_accelerator(self.engine)
            self.logger.info("Éter: Acelerador CUDA nativo ativado com sucesso")
        except Exception as e:
            self.logger.error(f"Éter: Erro ao ativar acelerador CUDA nativo: {str(e)}")

    # ... existing code ...

    def optimize_operations(self):
        """Otimiza operações do framework para máxima performance"""
        if self.engine is None:
            return

        # Verifica se kernels CUDA nativos estão disponíveis
        if self.cuda_accelerator:
            # Já ativado durante a inicialização
            report = self.cuda_accelerator.get_acceleration_report()
            self.logger.info(
                f"Éter: {report['active_accelerations']} operações aceleradas com CUDA nativo"
            )
            self.logger.info(f"Éter: Speedup médio de {report['average_speedup']:.2f}x")
        else:
            # Tenta otimizar usando meios alternativos
            self._optimize_with_fallback_methods()

    def _optimize_with_fallback_methods(self):
        """Otimiza operações usando métodos alternativos quando CUDA nativo não está disponível"""
        # Implementação de otimizações alternativas
        pass

    # ... existing code ...


# Instância global para uso direto
eter_bridge = EterFrameworkBridge()
