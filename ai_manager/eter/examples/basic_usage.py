#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemplo básico de uso do assistente Èter

Este script demonstra como inicializar e utilizar o assistente
Èter para processamento de linguagem natural e execução de tarefas.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any

# Adicionar diretório raiz ao path para importações
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from cerberus_api.pipeline_builder.ai_manager.eter import initialize, EterAgent
from cerberus_api.utils.logging_config import get_logger

# Configurar logger
logger = get_logger("eter_example")


def run_interactive_session(agent: EterAgent) -> None:
    """
    Executa uma sessão interativa com o Èter.

    Args:
        agent: Instância inicializada do EterAgent
    """
    print("\n" + "=" * 50)
    print("   SESSÃO INTERATIVA COM O ÈTER")
    print("=" * 50)
    print("\nDigite 'sair' para encerrar a sessão.")
    print("Digite 'ajuda' para ver comandos disponíveis.\n")

    while True:
        try:
            user_input = input("\nVocê: ")

            if user_input.lower() in ("sair", "exit", "quit"):
                print("\nEncerrando sessão. Até logo!\n")
                break

            if user_input.lower() == "ajuda":
                print("\nComandos disponíveis:")
                print("  ajuda     - Exibe esta ajuda")
                print("  sair      - Encerra a sessão")
                print("  executar  - Seguido por uma descrição de tarefa a executar")
                print("  Qualquer outro texto será interpretado como uma consulta\n")
                continue

            if user_input.lower().startswith("executar "):
                task = user_input[9:].strip()
                print(f"\nExecutando tarefa: {task}\n")

                results = agent.execute_task(task)
                print(f"\nResultado: {results['status']}")
                print(f"Mensagem: {results['message']}\n")
                continue

            # Processar input normal
            response = agent.process_input(user_input)
            print(f"\nÈter: {response}")

        except KeyboardInterrupt:
            print("\n\nSessão interrompida. Encerrando...")
            break
        except Exception as e:
            print(f"\nErro: {e}")


def demonstrate_api_usage(agent: EterAgent) -> None:
    """
    Demonstra o uso programático da API do Èter.

    Args:
        agent: Instância inicializada do EterAgent
    """
    print("\n" + "=" * 50)
    print("   DEMONSTRAÇÃO DE USO DA API")
    print("=" * 50)

    # Exemplo 1: Processando uma consulta simples
    query = "O que você pode fazer por mim?"
    print(f"\n1. Enviando consulta: '{query}'")
    response = agent.process_input(query)
    print(f"   Resposta: {response}")

    # Exemplo 2: Executando uma tarefa
    task = "listar os pipelines disponíveis"
    print(f"\n2. Executando tarefa: '{task}'")
    result = agent.execute_task(task)
    print(f"   Resultado: {result['status']}")
    print(f"   Mensagem: {result['message']}")

    # Exemplo 3: Observação e aprendizado
    print("\n3. Registrando observação para aprendizado")
    context = {"action": "create_pipeline", "parameters": {"name": "test_pipeline"}}
    result = {"status": "success", "pipeline_id": "12345"}

    agent.learn_from_observation(context, result)
    print("   Observação registrada com sucesso")

    print("\nDemonstração concluída!\n")


def main():
    """Função principal do exemplo."""
    parser = argparse.ArgumentParser(description="Exemplo de uso do Èter")

    parser.add_argument(
        "--model-dir", default="./models", help="Diretório para armazenar modelos"
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
        "--interactive",
        action="store_true",
        help="Iniciar sessão interativa após demonstração",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # Inicializar o Èter
        print(f"Inicializando Èter para usuário {args.user_id}...")
        agent = initialize(
            models_dir=args.model_dir,
            knowledge_dir=args.knowledge_dir,
            workspace_dir=args.workspace_dir,
            user_id=args.user_id,
            verbose=args.verbose,
        )

        # Executar demonstração de API
        demonstrate_api_usage(agent)

        # Se solicitado, executar sessão interativa
        if args.interactive:
            run_interactive_session(agent)

        # Salvar estado ao finalizar
        agent.save_state()
        print("Estado do Èter salvo. Exemplo concluído.")

    except Exception as e:
        logger.error(f"Erro durante execução do exemplo: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
