#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste do carregador nativo de modelos GGUF
"""

import os
import sys
import logging
import argparse
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gguf_test")

# Ajustar path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


def main():
    parser = argparse.ArgumentParser(description="Teste do carregador GGUF nativo")
    parser.add_argument(
        "--model", type=str, required=True, help="Caminho para o arquivo GGUF"
    )
    parser.add_argument(
        "--list-tensors", action="store_true", help="Listar tensores do modelo"
    )
    parser.add_argument(
        "--metadata", action="store_true", help="Mostrar metadados do modelo"
    )
    parser.add_argument("--prompt", type=str, help="Testar geração com prompt")

    args = parser.parse_args()

    try:
        from cerberus_api.pipeline_builder.ai_manager.native_loaders.gguf_loader import (
            GGUFModel,
            GGUFReader,
        )

        if not os.path.exists(args.model):
            print(f"Erro: Arquivo não encontrado: {args.model}")
            return

        print(f"Carregando modelo GGUF: {args.model}")

        if args.list_tensors or args.metadata:
            # Usar o leitor para leitura básica
            reader = GGUFReader(args.model)

            if args.metadata:
                print("\nMetadados do modelo:")
                metadata = reader.get_metadata()
                for key, value in metadata.items():
                    # Limitar exibição de arrays grandes
                    if isinstance(value, list) and len(value) > 10:
                        print(f"  {key}: [lista com {len(value)} itens]")
                    else:
                        print(f"  {key}: {value}")

            if args.list_tensors:
                print("\nTensores disponíveis:")
                tensor_names = reader.get_tensor_names()
                for name in tensor_names:
                    tensor_info = reader.tensors.get(name, {})
                    shape = tensor_info.get("shape", [])
                    shape_str = "x".join([str(s) for s in shape])
                    print(f"  {name}: {shape_str}")

        if args.prompt:
            # Carregar modelo completo para geração
            model = GGUFModel(args.model)

            print(f"\nPrompt: {args.prompt}")
            print("Gerando resposta...")

            response = model.generate(
                prompt=args.prompt, max_tokens=100, temperature=0.8, top_p=0.95
            )

            print("\nResposta:")
            print(response)

    except ImportError as e:
        print(f"Erro ao importar módulos: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")


if __name__ == "__main__":
    main()
