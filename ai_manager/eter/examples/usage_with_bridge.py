#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemplo de uso do Èter com a ponte para o framework CerBerus FMK.

Este exemplo demonstra como inicializar o Èter com a ponte do framework,
processar comandos em linguagem natural e acessar componentes do framework.
"""

import os
import sys
import time
import logging

# Configurar o path para incluir o diretório raiz do projeto
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
)

# Importar o Èter
from cerberus_api.pipeline_builder.ai_manager.eter import create_eter_instance

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"/tmp/eter_example_{int(time.time())}.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("eter_example")


def main():
    """Função principal que demonstra o uso do Èter com a ponte do framework."""

    logger.info("Inicializando o Èter com a ponte do framework...")

    # Criar uma instância do Èter com a ponte do framework ativada
    eter = create_eter_instance(
        enable_framework_bridge=True,
        register_components=True,
        model_name="Qwen2.5-7B-Instruct-1M-Q6_K",  # Substituir pelo nome do modelo disponível
    )

    # Exibir informações sobre os componentes registrados
    component_info = (
        eter.framework.discover_framework_capabilities()
        if hasattr(eter, "framework")
        else {}
    )
    logger.info(f"Componentes registrados: {component_info}")

    print("\n" + "=" * 80)
    print("Bem-vindo ao exemplo de uso do Èter com a ponte do framework CerBerus FMK!")
    print("=" * 80)
    print("\nEste exemplo demonstra as funcionalidades do Èter integrado ao framework.")
    print("Você pode experimentar os seguintes exemplos:\n")

    print("1. Execução de comandos de sistema:")
    print("   - Listando diretório: eter.execute_shell_command('ls -la')")
    print(
        "   - Verificando versão Python: eter.execute_shell_command('python --version')"
    )

    # Exemplo de execução de comando de sistema
    print("\nExecutando 'ls -la':")
    result = eter.execute_shell_command("ls -la")
    print(f"Resultado: {result['stdout']}")

    print("\n2. Processamento de comandos em linguagem natural:")
    print("   - eter.process_user_command('Analise o arquivo README.md')")
    print("   - eter.process_user_command('Crie um pipeline para análise de logs')")
    print("   - eter.process_input('Quais componentes estão disponíveis?')")

    # Exemplo de processamento de comando em linguagem natural
    print("\nProcessando comando 'Analise o arquivo README.md':")
    response = eter.process_user_command("Analise o arquivo README.md")
    print(f"Resposta: {response}")

    print("\n3. Acesso direto aos componentes do framework:")
    print(
        "   - Carregando modelo: eter.set_active_model('Qwen2.5-7B-Instruct-1M-Q6_K')"
    )
    print(
        "   - Memória: eter.memory_manager.add_memory('exemplo', 'Esta é uma memória de teste')"
    )
    print(
        "   - Pipeline: pipeline_builder = eter.get_framework_component('pipeline_builder')"
    )

    # Exemplo de acesso à memória
    eter.memory_manager.add_memory(
        memory_key="exemplo_teste",
        content="Esta é uma memória de teste criada durante o exemplo de uso do Èter.",
        metadata={"tipo": "exemplo", "prioridade": "alta"},
    )

    print("\nMemória 'exemplo_teste' adicionada. Recuperando:")
    memory = eter.memory_manager.get_memory("exemplo_teste")
    print(f"Conteúdo: {memory}")

    print("\n" + "=" * 80)
    print("Teste interativo: Digite comandos para o Èter (ou 'sair' para encerrar)")
    print("Use '!' no início para executar comandos de shell. Ex: !ls -la")
    print("=" * 80)

    while True:
        try:
            user_input = input("\nÈter> ")
            if user_input.lower() in ["sair", "exit", "quit"]:
                break

            # Processar comando de shell (com !)
            if user_input.startswith("!"):
                shell_cmd = user_input[1:].strip()
                if shell_cmd:
                    try:
                        print(f"Executando: {shell_cmd}")
                        result = eter.execute_shell_command(shell_cmd)
                        if result["success"]:
                            print(
                                f"\nComando executado em {result['elapsed_time']:.2f}s"
                            )
                            if result["stdout"]:
                                print(f"\n{result['stdout']}")
                            if result["stderr"]:
                                print(f"\nStderr:\n{result['stderr']}")
                        else:
                            print(
                                f"\nErro ao executar comando: {result.get('error', 'Erro desconhecido')}"
                            )
                    except Exception as e:
                        print(f"Erro: {e}")
            else:
                # Processar comando normal
                try:
                    start_time = time.time()
                    response = eter.process_input(user_input)
                    elapsed_time = time.time() - start_time
                    print(f"\nResposta ({elapsed_time:.2f}s):\n{response}")
                except Exception as e:
                    print(f"Erro ao processar comando: {e}")

        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário.")
            break
        except Exception as e:
            print(f"Erro: {e}")

    print("\nEncerrando exemplo de uso do Èter.")


if __name__ == "__main__":
    main()
