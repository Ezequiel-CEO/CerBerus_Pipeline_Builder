#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemplo de uso do EnhancedAnalyzer com gerenciamento de memória.

Este exemplo demonstra como o sistema processa grandes volumes de texto
de forma eficiente, utilizando o gerenciador de memória para otimizar
o uso de recursos.
"""

import os
import time
import logging
import random
from typing import List, Dict, Any

from cerberus_api.pipeline_builder.models import (
    ContentChunk,
    AnalysisResult,
    ContentType,
)
from cerberus_api.pipeline_builder.analyzers.enhanced_analyzer import EnhancedAnalyzer
from cerberus_api.pipeline_builder.utils.memory_manager import memory_manager
from cerberus_api.utils.logging_config import get_logger

# Configurar logger
logger = get_logger("memory_example")


def generate_test_data(
    num_chunks: int = 100, min_size: int = 500, max_size: int = 5000
) -> List[ContentChunk]:
    """
    Gera dados de teste aleatórios.

    Args:
        num_chunks: Número de chunks a gerar
        min_size: Tamanho mínimo em palavras
        max_size: Tamanho máximo em palavras

    Returns:
        Lista de chunks para teste
    """
    logger.info(f"Gerando {num_chunks} chunks de teste...")

    # Palavras de exemplo para gerar conteúdo aleatório
    words = [
        "inteligência",
        "artificial",
        "aprendizado",
        "máquina",
        "dados",
        "análise",
        "algoritmo",
        "modelo",
        "treinamento",
        "processamento",
        "linguagem",
        "natural",
        "reconhecimento",
        "padrões",
        "redes",
        "neurais",
        "transformers",
        "embedding",
        "classificação",
        "regressão",
        "clustering",
        "otimização",
        "validação",
        "teste",
        "acurácia",
        "precisão",
        "recall",
        "f1",
        "erro",
        "viés",
        "variância",
        "hiperparâmetros",
        "tensor",
        "vetor",
        "matriz",
        "gradiente",
        "função",
        "custo",
        "perda",
        "ativação",
        "dropout",
        "regularização",
        "normalização",
        "batch",
        "época",
        "iteração",
    ]

    # Frases de exemplo para gerar parágrafos mais realistas
    templates = [
        "A {0} é fundamental para o desenvolvimento de sistemas de {1}.",
        "Os modelos de {0} permitem melhor {1} de {2}.",
        "Utilizando {0}, podemos obter resultados superiores em tarefas de {1}.",
        "A pesquisa em {0} avançou significativamente nos últimos anos, especialmente em {1}.",
        "Uma das limitações da {0} é a necessidade de grandes volumes de {1} para {2}.",
        "O processo de {0} inclui etapas como {1}, {2} e {3}.",
        "A performance de sistemas baseados em {0} depende fortemente da qualidade dos {1}.",
        "Métodos de {0} podem ser aplicados em diversos contextos, como {1} e {2}.",
        "A evolução das técnicas de {0} revolucionou a forma como abordamos problemas de {1}.",
        "Combinando {0} com {1}, conseguimos resultados promissores em {2}.",
    ]

    # Gerar chunks aleatórios
    chunks = []
    for i in range(num_chunks):
        # Gerar tamanho aleatório para o chunk
        size = random.randint(min_size, max_size)

        # Gerar conteúdo
        paragraphs = []
        words_count = 0

        while words_count < size:
            # Número de frases no parágrafo
            num_sentences = random.randint(3, 10)
            paragraph = []

            for _ in range(num_sentences):
                template = random.choice(templates)
                sentence = template.format(
                    random.choice(words),
                    random.choice(words),
                    random.choice(words),
                    random.choice(words),
                )
                paragraph.append(sentence)
                words_count += len(sentence.split())

            paragraphs.append(" ".join(paragraph))

            # Adicionar alguns parágrafos apenas com palavras para diversificar
            if random.random() < 0.2:
                word_paragraph = " ".join(
                    random.choices(words, k=random.randint(20, 50))
                )
                paragraphs.append(word_paragraph)
                words_count += len(word_paragraph.split())

        # Montar o texto completo
        text = "\n\n".join(paragraphs)

        # Criar chunk
        chunk = ContentChunk(
            id=f"chunk_{i}",
            task_id="example_task",
            content_type=ContentType.TEXT,
            text=text,
            source_location=f"example/location/{i}",
            length=len(text),
        )

        chunks.append(chunk)

    logger.info(
        f"Gerados {len(chunks)} chunks com tamanho total: {sum(len(c.text) for c in chunks)/(1024*1024):.2f} MB"
    )
    return chunks


def run_analysis_demo():
    """Executa o demo de análise com gerenciamento de memória."""
    logger.info("Iniciando demo de gerenciamento de memória...")

    # Criar diretório para resultados
    output_dir = "output/memory_demo"
    os.makedirs(output_dir, exist_ok=True)

    # Configurar memória ao iniciar
    memory_manager.log_memory_usage("início do demo")

    # Configurar analisador
    analyzer = EnhancedAnalyzer(
        storage_dir=output_dir,
        stream_processing=True,  # Habilitar processamento em streaming
        memory_efficient=True,  # Usar modo eficiente em memória
        batch_size=10,  # Tamanho inicial do lote
        use_gpu=True,  # Tentar usar GPU quando disponível
    )

    logger.info(f"EnhancedAnalyzer configurado com dispositivo: {analyzer.device}")

    # Gerar dados de teste - 3 conjuntos de tamanhos diferentes
    small_data = generate_test_data(10, 100, 500)  # Pequeno: ~10 chunks
    medium_data = generate_test_data(50, 500, 3000)  # Médio: ~50 chunks
    large_data = generate_test_data(200, 1000, 8000)  # Grande: ~200 chunks

    # Processar conjunto pequeno
    logger.info("Processando conjunto PEQUENO de dados...")
    start_time = time.time()
    small_results = analyzer.analyze(small_data)
    logger.info(
        f"Conjunto PEQUENO processado em {time.time() - start_time:.2f} segundos"
    )
    logger.info(f"Resultados: {len(small_results)} chunks analisados")
    memory_manager.log_memory_usage("após conjunto pequeno")

    # Processar conjunto médio
    logger.info("Processando conjunto MÉDIO de dados...")
    start_time = time.time()
    medium_results = analyzer.analyze(medium_data)
    logger.info(f"Conjunto MÉDIO processado em {time.time() - start_time:.2f} segundos")
    logger.info(f"Resultados: {len(medium_results)} chunks analisados")
    memory_manager.log_memory_usage("após conjunto médio")

    # Processar conjunto grande
    logger.info("Processando conjunto GRANDE de dados...")
    start_time = time.time()
    large_results = analyzer.analyze(large_data)
    logger.info(
        f"Conjunto GRANDE processado em {time.time() - start_time:.2f} segundos"
    )
    logger.info(f"Resultados: {len(large_results)} chunks analisados")
    memory_manager.log_memory_usage("após conjunto grande")

    # Exibir estatísticas
    logger.info("Estatísticas de relevância:")
    logger.info(
        f"Conjunto pequeno - média: {sum(r.metadata.get('enhanced_relevance', 0) for r in small_results) / len(small_results):.4f}"
    )
    logger.info(
        f"Conjunto médio - média: {sum(r.metadata.get('enhanced_relevance', 0) for r in medium_results) / len(medium_results):.4f}"
    )
    logger.info(
        f"Conjunto grande - média: {sum(r.metadata.get('enhanced_relevance', 0) for r in large_results) / len(large_results):.4f}"
    )

    # Limpar memória ao finalizar
    memory_manager.clear_memory()
    memory_manager.log_memory_usage("fim do demo")

    logger.info("Demo de gerenciamento de memória concluído com sucesso!")


if __name__ == "__main__":
    run_analysis_demo()
