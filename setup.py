#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de setup para o PipelineBuilder.

Este script configura o ambiente de execução do PipelineBuilder,
criando diretórios necessários e verificando dependências.
"""

import os
import sys
import subprocess
import argparse
import shutil
import logging
from pathlib import Path

# Adicionar diretório pai ao path para importação
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from cerberus_api.utils.logging_config import APILogger

    logger = APILogger("setup")
except ImportError:
    # Fallback para logging padrão caso o módulo ainda não esteja configurado
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("setup")


def create_directories(base_dir="data"):
    """
    Cria os diretórios necessários para o PipelineBuilder.

    Args:
        base_dir: Diretório base para armazenamento.
    """
    logger.info(f"Criando diretórios em {base_dir}")

    # Diretórios para armazenamento de dados
    dirs = [
        Path(base_dir) / "pipelines",
        Path(base_dir) / "web_content",
        Path(base_dir) / "pdf_content",
        Path(base_dir) / "github_content",
        Path(base_dir) / "youtube_content",
        Path(base_dir) / "processed",
        Path(base_dir) / "approved",
        Path("logs"),
    ]

    # Criar cada diretório
    for d in dirs:
        logger.info(f"Criando diretório: {d}")
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Diretórios criados com sucesso")


def check_dependencies():
    """
    Verifica se as dependências estão instaladas.

    Returns:
        Lista de dependências faltantes.
    """
    logger.info("Verificando dependências")

    dependencies = [
        "pydantic",
        "requests",
        "beautifulsoup4",
        "nltk",
        "spacy",
        "scikit-learn",
        "PyPDF2",
        "pdfminer.six",
        "PyMuPDF",
        "GitPython",
        "gradio",
    ]

    missing = []

    for dep in dependencies:
        try:
            __import__(dep.split(".")[0])
            logger.info(f"✓ {dep} instalado")
        except ImportError:
            logger.warning(f"✗ {dep} não encontrado")
            missing.append(dep)

    return missing


def install_dependencies(missing):
    """
    Instala dependências faltantes.

    Args:
        missing: Lista de dependências para instalar.

    Returns:
        True se todas as instalações forem bem-sucedidas.
    """
    if not missing:
        logger.info("Todas as dependências já estão instaladas")
        return True

    logger.info(f"Instalando {len(missing)} dependências")

    for dep in missing:
        logger.info(f"Instalando {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            logger.info(f"✓ {dep} instalado com sucesso")
        except subprocess.CalledProcessError:
            logger.error(f"✗ Falha ao instalar {dep}")
            return False

    return True


def download_nltk_data():
    """
    Baixa os dados necessários para o NLTK.

    Returns:
        True se o download for bem-sucedido.
    """
    logger.info("Baixando dados do NLTK")

    try:
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        logger.info("✓ Dados do NLTK baixados com sucesso")
        return True
    except Exception as e:
        logger.error(f"✗ Falha ao baixar dados do NLTK: {e}")
        return False


def download_spacy_models():
    """
    Baixa os modelos necessários para o spaCy.

    Returns:
        True se o download for bem-sucedido.
    """
    logger.info("Baixando modelos do spaCy")

    models = ["pt_core_news_sm", "en_core_web_sm"]
    success = True

    for model in models:
        logger.info(f"Baixando modelo {model}...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", model],
                capture_output=True,
                text=True,
            )
            if (
                "already installed" in result.stdout
                or "Successfully installed" in result.stdout
            ):
                logger.info(f"✓ Modelo {model} pronto")
            else:
                logger.warning(
                    f"? Status desconhecido para modelo {model}: {result.stdout}"
                )
        except Exception as e:
            logger.error(f"✗ Falha ao baixar modelo {model}: {e}")
            success = False

    return success


def setup_example_files():
    """
    Configura arquivos de exemplo.

    Returns:
        True se a configuração for bem-sucedida.
    """
    logger.info("Verificando arquivos de exemplo")

    example_files = [
        "examples/simple_pipeline.py",
        "examples/multi_source_pipeline.py",
        "examples/sources_example.json",
    ]

    success = True
    base_path = os.path.dirname(os.path.abspath(__file__))

    for file in example_files:
        full_path = os.path.join(base_path, file)
        if os.path.exists(full_path):
            logger.info(f"✓ Arquivo {file} encontrado")
        else:
            logger.warning(f"✗ Arquivo {file} não encontrado")
            success = False

    return success


def parse_arguments():
    """
    Analisa os argumentos da linha de comando.

    Returns:
        Namespace com os argumentos parseados.
    """
    parser = argparse.ArgumentParser(description="Setup para o PipelineBuilder")
    parser.add_argument(
        "--base-dir", default="data", help="Diretório base para armazenamento"
    )
    parser.add_argument(
        "--skip-deps", action="store_true", help="Pular verificação de dependências"
    )
    parser.add_argument(
        "--skip-nltk", action="store_true", help="Pular download de dados do NLTK"
    )
    parser.add_argument(
        "--skip-spacy", action="store_true", help="Pular download de modelos do spaCy"
    )
    return parser.parse_args()


def main():
    """
    Função principal do script de setup.
    """
    print("=" * 80)
    print(" CerBerus.AI - PipelineBuilder Setup")
    print("=" * 80)
    print()

    args = parse_arguments()

    # Criar diretórios
    create_directories(args.base_dir)

    # Verificar dependências
    if not args.skip_deps:
        missing = check_dependencies()
        if missing:
            success = install_dependencies(missing)
            if not success:
                logger.error("Falha ao instalar dependências")
                print("\nAlgumas dependências não puderam ser instaladas.")
                print("Por favor, instale manualmente com:")
                print(f"pip install {' '.join(missing)}")
                return False
    else:
        logger.info("Verificação de dependências ignorada")

    # Baixar dados do NLTK
    if not args.skip_nltk:
        nltk_success = download_nltk_data()
        if not nltk_success:
            logger.warning("Falha ao baixar dados do NLTK")
    else:
        logger.info("Download de dados do NLTK ignorado")

    # Baixar modelos do spaCy
    if not args.skip_spacy:
        spacy_success = download_spacy_models()
        if not spacy_success:
            logger.warning("Falha ao baixar alguns modelos do spaCy")
    else:
        logger.info("Download de modelos do spaCy ignorado")

    # Verificar arquivos de exemplo
    example_success = setup_example_files()
    if not example_success:
        logger.warning("Alguns arquivos de exemplo não foram encontrados")

    print("\n" + "=" * 80)
    print(" Setup concluído!")
    print("=" * 80)
    print("\nPara começar, execute um dos exemplos:")
    print(
        f"python -m cerberus_api.pipeline_builder.cli example --type simple --base-dir {args.base_dir}"
    )
    print(f"\nOu inicie o dashboard:")
    print(
        f"python -m cerberus_api.pipeline_builder.cli dashboard --base-dir {args.base_dir}"
    )
    print("\nConsulte a documentação para mais opções.")

    return True


if __name__ == "__main__":
    main()
