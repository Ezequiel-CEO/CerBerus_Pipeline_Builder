#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstração de uso do sistema de modelos nativos do CerBerusFMK
"""

import os
import sys
import logging
import argparse

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("demo")

# Ajustar path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


def main():
    parser = argparse.ArgumentParser(
        description="Demonstração de modelos nativos do CerBerusFMK"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Caminho para o modelo ou nome do modelo na pasta models/",
    )
    parser.add_argument(
        "--prompt", type=str, default="Olá, como você está?", help="Prompt inicial"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Número máximo de tokens a gerar"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperatura para geração"
    )
    parser.add_argument("--cpu", action="store_true", help="Forçar uso de CPU")
    parser.add_argument(
        "--list-models", action="store_true", help="Listar modelos disponíveis"
    )
    parser.add_argument(
        "--create-template", type=str, help="Criar template para um novo modelo"
    )

    args = parser.parse_args()

    try:
        from cerberus_api.pipeline_builder.ai_manager.raw_gpu import (
            create_optimized_engine,
            list_available_models,
            create_model_template,
        )

        # Listar modelos disponíveis
        if args.list_models:
            models = list_available_models()
            if models:
                print("Modelos disponíveis:")
                for model in models:
                    print(f"  - {model}")
            else:
                print(
                    "Nenhum modelo encontrado. Use --create-template para criar um template."
                )
            return

        # Criar template de modelo
        if args.create_template:
            path = create_model_template(args.create_template)
            print(f"Template de modelo criado em: {path}")
            print(
                f"Para usar, coloque seu arquivo de modelo na pasta e execute: --model {args.create_template}"
            )
            return

        # Verificar se modelo foi especificado
        if not args.model:
            print(
                "Erro: Você deve especificar um modelo com --model ou listar modelos com --list-models"
            )
            return

        # Carregar modelo
        print(f"Carregando modelo: {args.model}")
        engine = create_optimized_engine(
            model_path=args.model, use_gpu=not args.cpu, n_ctx=2048
        )

        if not engine:
            print("Erro: Falha ao carregar modelo")
            return

        # Gerar texto
        print(f"\nPrompt: {args.prompt}")
        print("Gerando resposta...")

        response = engine.generate(
            prompt=args.prompt, max_tokens=args.max_tokens, temperature=args.temperature
        )

        print("\nResposta:")
        print(response)

    except ImportError as e:
        print(f"Erro ao importar módulos: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")


if __name__ == "__main__":
    main()
