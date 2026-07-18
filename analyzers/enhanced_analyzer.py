#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analisador de conteúdo avançado para o PipelineBuilder.

Implementa técnicas mais sofisticadas para análise de relevância
como embeddings e similaridade semântica.
"""

import os
import re
import json
import hashlib
import logging
import math
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from collections import Counter
import uuid

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from cerberus_api.pipeline_builder.models import (
    ContentChunk,
    AnalysisResult,
    AnalysisTask,
    ContentCategory,
    TaskStatus,
)
from cerberus_api.pipeline_builder.analyzers.content_analyzer import ContentAnalyzer
from cerberus_api.utils.logging_config import APILogger
from cerberus_api.utils.logging_config import get_logger
from cerberus_api.pipeline_builder.utils.memory_manager import memory_manager
from cerberus_api.pipeline_builder.ai_manager.eter import EterAgent
from cerberus_api.pipeline_builder.ai_manager.eter.interface import EterCLI

# Configurar logger
logger = APILogger("enhanced_analyzer")


class EnhancedAnalyzer(ContentAnalyzer):
    """
    Analisador de conteúdo avançado com recursos de NLP.
    """

    def __init__(
        self,
        min_relevance_score: float = 0.5,
        keywords_ratio: float = 0.05,
        stopwords_langs: List[str] = ["portuguese", "english"],
        storage_dir: Optional[str] = None,
        spacy_model: str = "pt_core_news_sm",
        use_embeddings: bool = True,
        domain_keywords: Optional[List[str]] = None,
        calculate_similarity: bool = False,
        generate_summary: bool = False,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 10,
        summary_algorithm: str = "textrank",  # 'textrank' ou 'tfidf'
        max_text_length: int = 50000,  # Limite máximo de texto para processar de uma vez
        stream_processing: bool = True,  # Habilitar processamento em streaming
        memory_efficient: bool = True,  # Usar modo eficiente em memória
        model_name: str = "pt_core_news_lg",
        similarity_threshold: float = 0.75,
        relevance_threshold: float = 0.60,
        use_gpu: bool = True,
    ):
        """
        Inicializa o analisador avançado.

        Args:
            min_relevance_score: Pontuação mínima de relevância.
            keywords_ratio: Proporção de palavras-chave.
            stopwords_langs: Idiomas para stopwords.
            storage_dir: Diretório para armazenamento dos resultados.
            spacy_model: Modelo do spaCy a ser usado.
            use_embeddings: Se deve usar embeddings para análise.
            domain_keywords: Palavras-chave do domínio para análise de relevância.
            calculate_similarity: Se deve calcular similaridade entre chunks.
            generate_summary: Se deve gerar resumos automáticos do conteúdo.
            embedding_model: Modelo para geração de embeddings (sentence-transformers).
            batch_size: Tamanho do lote para processamento em paralelo.
            summary_algorithm: Algoritmo para geração de resumos ('textrank' ou 'tfidf').
            max_text_length: Tamanho máximo de texto a processar de uma vez.
            stream_processing: Se deve processar em modo streaming.
            memory_efficient: Se deve usar modo de economia de memória.
            model_name: Nome do modelo spaCy a ser utilizado
            similarity_threshold: Limiar mínimo para considerar textos similares
            relevance_threshold: Limiar mínimo para considerar texto relevante
            use_gpu: Se deve tentar usar GPU quando disponível
        """
        super().__init__(
            min_relevance_score=min_relevance_score,
            keywords_ratio=keywords_ratio,
            stopwords_langs=stopwords_langs,
            storage_dir=storage_dir,
            extract_entities=True,
            extract_sentiment=True,
            nlp_model=spacy_model,
        )

        self.use_embeddings = use_embeddings
        self.spacy_model = spacy_model
        self.domain_keywords = domain_keywords or []
        self.calculate_similarity = calculate_similarity
        self.generate_summary = generate_summary
        self.embedding_model = embedding_model
        self.summary_algorithm = summary_algorithm
        self.batch_size = batch_size
        self.max_text_length = max_text_length
        self.stream_processing = stream_processing
        self.memory_efficient = memory_efficient

        # TF-IDF para análise de relevância
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 2)
        )

        # Inicializar modelo de embeddings se necessário
        self.sentence_transformer = None
        if self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                self.sentence_transformer = SentenceTransformer(self.embedding_model)
                logger.info(
                    f"Modelo de embeddings '{self.embedding_model}' carregado com sucesso"
                )
            except ImportError:
                logger.warning(
                    "sentence-transformers não está instalado. Para usar embeddings, instale com: pip install sentence-transformers"
                )
                self.use_embeddings = False
            except Exception as e:
                logger.warning(f"Erro ao carregar modelo de embeddings: {e}")
                self.use_embeddings = False

        # Carregar modelo spaCy se disponível
        self.nlp = None
        if SPACY_AVAILABLE and self.use_embeddings:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Modelo spaCy '{spacy_model}' carregado com sucesso")
            except Exception as e:
                logger.warning(f"Erro ao carregar modelo spaCy: {e}")
                logger.warning(
                    "Para instalar o modelo, execute: python -m spacy download pt_core_news_sm"
                )
                self.use_embeddings = False
        else:
            logger.warning("spaCy não está disponível, embeddings não serão usados")
            self.use_embeddings = False

        # Configurar memória e GPU
        memory_manager.enable_cuda = use_gpu
        memory_manager.enable_streaming = stream_processing
        memory_manager.max_batch_size = batch_size

        self.device = memory_manager.device
        logger.info(f"EnhancedAnalyzer inicializado com dispositivo: {self.device}")

        # Cache para embeddings
        self.embeddings_cache = {}

        # Registrar uso inicial de memória
        memory_manager.log_memory_usage("inicialização do EnhancedAnalyzer")

    def analyze(
        self, chunks: List[ContentChunk], storage_dir: Optional[str] = None
    ) -> List[AnalysisResult]:
        """
        Analisa uma lista de chunks de conteúdo, gerando metadados avançados.
        Utiliza processamento em streaming e otimização de memória para lidar com grandes volumes.

        Args:
            chunks: Lista de chunks de conteúdo para análise
            storage_dir: Diretório para armazenar os resultados da análise

        Returns:
            Lista de resultados de análise com metadados
        """
        if not chunks:
            logger.warning("Nenhum chunk fornecido para análise")
            return []

        # Inicializar lista de resultados
        results = []

        # Configurar diretório de armazenamento
        if storage_dir:
            self.storage_dir = storage_dir

        # Registrar início da análise
        logger.info(f"Iniciando análise de {len(chunks)} chunks com EnhancedAnalyzer")
        memory_manager.log_memory_usage("início de análise")

        # Calcular tamanho dos dados para decisão de streaming
        total_size = sum(
            len(chunk.text.encode("utf-8")) for chunk in chunks if chunk.text
        )
        item_sizes = [len(chunk.text.encode("utf-8")) for chunk in chunks if chunk.text]

        # Otimizar tamanho do batch com base na memória disponível
        self.batch_size = memory_manager.optimize_batch_size(len(chunks))

        # Decidir entre processamento em lote ou streaming
        use_streaming = (
            self.stream_processing
            and memory_manager.should_process_in_streaming(total_size, item_sizes)
        )

        if use_streaming:
            logger.info(
                f"Usando processamento em streaming para {len(chunks)} chunks (tamanho total: {total_size/(1024*1024):.2f} MB)"
            )
            results = self._process_in_streaming(chunks)
        else:
            logger.info(
                f"Usando processamento em lote para {len(chunks)} chunks com tamanho de lote {self.batch_size}"
            )
            results = self._process_in_batches(chunks)

        # Registrar fim da análise
        logger.info(
            f"Análise concluída para {len(chunks)} chunks. Resultados gerados: {len(results)}"
        )
        memory_manager.log_memory_usage("fim de análise")

        return results

    def _process_in_batches(self, chunks: List[ContentChunk]) -> List[AnalysisResult]:
        """
        Processa chunks em lotes para otimizar uso de memória.

        Args:
            chunks: Lista de chunks para processar

        Returns:
            Lista de resultados de análise
        """
        results = []

        # Dividir em lotes
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            logger.info(
                f"Processando lote {i//self.batch_size + 1}/{(len(chunks) + self.batch_size - 1)//self.batch_size} ({len(batch)} chunks)"
            )

            # Analisar cada chunk no lote
            batch_results = []
            for chunk in batch:
                try:
                    # Aplicar análise base para extrair metadados iniciais
                    base_metadata = self._extract_base_metadata(chunk.text)

                    # Aplicar análises avançadas
                    enhanced_metadata = self._enhance_metadata(
                        chunk.text, base_metadata
                    )

                    # Criar resultado da análise
                    result = AnalysisResult(
                        chunk_id=chunk.id,
                        task_id="test_task",
                        metadata=enhanced_metadata,
                        status=TaskStatus.COMPLETED,
                    )

                    batch_results.append(result)

                except Exception as e:
                    logger.error(f"Erro ao analisar chunk {chunk.id}: {e}")
                    batch_results.append(
                        AnalysisResult(
                            chunk_id=chunk.id,
                            task_id="test_task",
                            metadata={},
                            status=TaskStatus.FAILED,
                            error=str(e),
                        )
                    )

            # Adicionar resultados do lote
            results.extend(batch_results)

            # Limpar memória após cada lote
            if self.memory_efficient:
                memory_manager.clear_memory()

        return results

    def _process_in_streaming(self, chunks: List[ContentChunk]) -> List[AnalysisResult]:
        """
        Processa chunks em modo streaming para minimizar uso de memória.
        Adequado para grandes volumes de dados.

        Args:
            chunks: Lista de chunks para processar

        Returns:
            Lista de resultados de análise
        """
        results = []

        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"Processando chunk {i+1}/{len(chunks)} (id: {chunk.id})")

                # Verificar tamanho do chunk
                chunk_size = len(chunk.text.encode("utf-8")) if chunk.text else 0
                chunk_size_mb = chunk_size / (1024 * 1024)

                # Chunks muito grandes são processados em partes
                if chunk_size_mb > 5.0:  # Mais de 5MB
                    logger.info(
                        f"Chunk grande detectado ({chunk_size_mb:.2f} MB). Processando em modo especial."
                    )
                    result = self._process_large_chunk_in_streaming(chunk)
                else:
                    # Chunks normais
                    base_metadata = self._extract_base_metadata(chunk.text)
                    enhanced_metadata = self._enhance_metadata(
                        chunk.text, base_metadata
                    )

                    result = AnalysisResult(
                        chunk_id=chunk.id,
                        task_id="test_task",
                        metadata=enhanced_metadata,
                        status=TaskStatus.COMPLETED,
                    )

                results.append(result)

                # Limpar memória após cada chunk se for grande
                if chunk_size_mb > 1.0 and self.memory_efficient:
                    memory_manager.clear_memory()

                # Limpar memória periodicamente
                if i % 5 == 0 and i > 0 and self.memory_efficient:
                    memory_manager.clear_memory()

            except Exception as e:
                logger.error(f"Erro ao processar chunk {chunk.id} em streaming: {e}")
                results.append(
                    AnalysisResult(
                        chunk_id=chunk.id,
                        task_id="test_task",
                        metadata={},
                        status=TaskStatus.FAILED,
                        error=str(e),
                    )
                )

        return results

    def _process_large_chunk_in_streaming(self, chunk: ContentChunk) -> AnalysisResult:
        """
        Processa um chunk muito grande em modo de streaming especializado.
        Divide o texto em seções menores, processa cada uma e combina os resultados.

        Args:
            chunk: Chunk grande para processar

        Returns:
            Resultado da análise combinada
        """
        import nltk
        from collections import Counter

        logger.info(f"Dividindo chunk grande {chunk.id} em seções menores")

        try:
            # Garantir que temos o tokenizador de sentenças
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

            # Dividir em parágrafos/seções
            paragraphs = chunk.text.split("\n\n")

            # Se ainda temos parágrafos muito longos, subdividir
            sections = []
            for para in paragraphs:
                if len(para) > 5000:  # Parágrafos com mais de 5000 caracteres
                    # Dividir em sentenças e reagrupar
                    sentences = nltk.sent_tokenize(para)
                    current_section = ""
                    for sent in sentences:
                        if len(current_section) + len(sent) > 4000:
                            sections.append(current_section)
                            current_section = sent
                        else:
                            current_section += " " + sent
                    if current_section:
                        sections.append(current_section)
                else:
                    sections.append(para)

            logger.info(f"Chunk dividido em {len(sections)} seções menores")

            # Analisar cada seção
            section_results = []
            combined_metadata = {
                "entities": [],
                "keywords": [],
                "sentiment_score": 0.0,
                "relevance_score": 0.0,
                "language": "",
                "word_count": 0,
                "sentence_count": 0,
                "summary": "",
                "enhanced_relevance": 0.0,
            }

            language_votes = Counter()

            for i, section in enumerate(sections):
                if not section.strip():
                    continue

                logger.info(
                    f"Analisando seção {i+1}/{len(sections)} do chunk {chunk.id}"
                )

                # Extrair metadados da seção
                base_metadata = self._extract_base_metadata(section)

                # Para algumas análises avançadas, processamos todas as seções e combinamos
                language_votes[base_metadata.get("language", "")] += 1
                combined_metadata["word_count"] += base_metadata.get("word_count", 0)
                combined_metadata["sentence_count"] += base_metadata.get(
                    "sentence_count", 0
                )
                combined_metadata["relevance_score"] += base_metadata.get(
                    "relevance_score", 0
                ) * len(section)
                combined_metadata["sentiment_score"] += base_metadata.get(
                    "sentiment_score", 0
                ) * len(section)

                # Acumular entidades e palavras-chave
                combined_metadata["entities"].extend(base_metadata.get("entities", []))
                combined_metadata["keywords"].extend(base_metadata.get("keywords", []))

                # Limpar memória após cada seção
                if self.memory_efficient and i % 3 == 0 and i > 0:
                    memory_manager.clear_memory()

                # Guardar a seção para resumo se for relevante
                if base_metadata.get("relevance_score", 0) > self.relevance_threshold:
                    section_results.append((section, base_metadata))

            # Normalizar pontuações baseadas no tamanho
            total_length = len(chunk.text)
            if total_length > 0:
                combined_metadata["relevance_score"] /= total_length
                combined_metadata["sentiment_score"] /= total_length

            # Determinar idioma mais comum
            combined_metadata["language"] = (
                language_votes.most_common(1)[0][0] if language_votes else "unknown"
            )

            # Consolidar entidades e palavras-chave (remover duplicatas, limitar quantidade)
            if combined_metadata["entities"]:
                # Filtrar entidades únicas por texto
                unique_entities = {}
                for entity in combined_metadata["entities"]:
                    text = entity.get("text", "")
                    if text and (
                        text not in unique_entities
                        or entity.get("relevance", 0)
                        > unique_entities[text].get("relevance", 0)
                    ):
                        unique_entities[text] = entity
                combined_metadata["entities"] = list(unique_entities.values())

                # Limitar a 30 entidades mais relevantes
                combined_metadata["entities"] = sorted(
                    combined_metadata["entities"],
                    key=lambda x: x.get("relevance", 0),
                    reverse=True,
                )[:30]

            if combined_metadata["keywords"]:
                # Contar frequência de palavras-chave
                keyword_counter = Counter()
                for kw in combined_metadata["keywords"]:
                    keyword_counter[kw] += 1

                # Selecionar as 20 mais frequentes
                combined_metadata["keywords"] = [
                    kw for kw, _ in keyword_counter.most_common(20)
                ]

            # Calcular relevância aprimorada
            combined_metadata["enhanced_relevance"] = (
                self._calculate_enhanced_relevance(chunk.text)
            )

            # Gerar resumo a partir das seções mais relevantes
            # Ordenar seções por relevância
            section_results.sort(
                key=lambda x: x[1].get("relevance_score", 0), reverse=True
            )

            # Usar as 3-5 seções mais relevantes para o resumo
            top_sections = [
                section
                for section, _ in section_results[: min(5, len(section_results))]
            ]
            if top_sections:
                combined_text = "\n\n".join(top_sections)
                combined_metadata["summary"] = self._generate_enhanced_summary(
                    combined_text
                )

            return AnalysisResult(
                chunk_id=chunk.id,
                task_id="test_task",
                metadata=combined_metadata,
                status=TaskStatus.COMPLETED,
            )

        except Exception as e:
            logger.error(f"Erro ao processar chunk grande {chunk.id}: {e}")
            # Tentar uma abordagem mais simples em caso de erro
            try:
                basic_metadata = self._extract_base_metadata(
                    chunk.text[:10000]
                )  # Analisar apenas o início
                return AnalysisResult(
                    chunk_id=chunk.id,
                    task_id="test_task",
                    metadata=basic_metadata,
                    status=TaskStatus.PARTIALLY_COMPLETED,
                    error=f"Processamento parcial devido a erro: {e}",
                )
            except:
                return AnalysisResult(
                    chunk_id=chunk.id,
                    task_id="test_task",
                    metadata={},
                    status=TaskStatus.FAILED,
                    error=str(e),
                )

    def _calculate_enhanced_relevance(self, text: str) -> float:
        """
        Calcula a relevância aprimorada com base em múltiplos fatores.

        Args:
            text: Texto para análise

        Returns:
            Pontuação de relevância aprimorada entre 0 e 1
        """
        # Verificar se o texto é válido
        if not text or not text.strip():
            return 0.0

        # Se metadata não foi fornecido, extrair metadados básicos
        base_metadata = self._extract_base_metadata(text)

        # Obter dados do metadata
        word_count = base_metadata.get("word_count", 0)
        sentence_count = base_metadata.get("sentence_count", 0)
        language = base_metadata.get("language", "unknown")
        base_relevance = base_metadata.get("relevance_score", 0.0)
        keywords = base_metadata.get("keywords", [])
        entities = base_metadata.get("entities", [])

        # 1. Fator de qualidade do texto
        quality_score = 0.0

        # Densidade de informação (palavras por sentença)
        info_density = 0.0
        if sentence_count > 0:
            words_per_sentence = word_count / sentence_count
            # Normalizar para diferentes idiomas
            if language in ["english", "en", "portuguese", "pt"]:
                target_density = 20.0  # Alvo para inglês/português
            elif language in ["german", "de", "dutch", "nl"]:
                target_density = 25.0  # Alvo para alemão/holandês
            elif language in ["japanese", "ja", "chinese", "zh"]:
                target_density = 15.0  # Alvo para japonês/chinês
            else:
                target_density = 20.0  # Valor padrão

            info_density = min(1.0, words_per_sentence / target_density)

        # 2. Fator de comprimento
        # Penalizar textos muito curtos, valorizar textos de tamanho médio
        length_factor = 0.0
        if word_count > 0:
            # Função que valoriza textos na faixa de 100-2000 palavras
            length_factor = (
                min(1.0, word_count / 500)
                if word_count < 500
                else min(1.0, 4000 / max(word_count, 1))
            )

        # 3. Fator de informação (palavras-chave e entidades)
        info_factor = 0.0
        if word_count > 0:
            # Densidade de palavras-chave e entidades
            keyword_density = len(keywords) / max(1, word_count / 100)
            entity_density = len(entities) / max(1, word_count / 200)

            info_factor = min(1.0, (keyword_density * 0.6 + entity_density * 0.4))

        # 4. Fator de embedding (se disponível)
        embedding_factor = 0.0
        if self.sentence_transformer:
            try:
                # Usar embedding para verificar coerência
                embedding = self._generate_embedding(text)
                if embedding is not None:
                    # Valor entre 0.4 e 0.7 para evitar peso excessivo deste fator
                    embedding_factor = (
                        0.4 + (sum([abs(x) for x in embedding[:10]]) / 10) * 0.3
                    )
            except Exception as e:
                logger.warning(f"Erro ao calcular fator de embedding: {e}")

        # Pesos para cada fator
        weights = {
            "base_relevance": 0.35,
            "info_density": 0.15,
            "length_factor": 0.15,
            "info_factor": 0.25,
            "embedding_factor": 0.10,
        }

        # Se embeddings não estiverem disponíveis, redistribuir o peso
        if not self.sentence_transformer:
            weights["base_relevance"] += weights["embedding_factor"] * 0.5
            weights["info_factor"] += weights["embedding_factor"] * 0.5
            weights["embedding_factor"] = 0.0

        # Calcular pontuação final
        enhanced_relevance = (
            base_relevance * weights["base_relevance"]
            + info_density * weights["info_density"]
            + length_factor * weights["length_factor"]
            + info_factor * weights["info_factor"]
            + embedding_factor * weights["embedding_factor"]
        )

        # Garantir que o resultado está entre 0 e 1
        return min(1.0, max(0.0, enhanced_relevance))

    def _enhance_metadata(
        self, text: str, base_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aprimora os metadados básicos com análises avançadas.

        Args:
            text: Texto para análise
            base_metadata: Metadados básicos já extraídos

        Returns:
            Metadados aprimorados
        """
        # Iniciar com os metadados básicos
        enhanced_metadata = dict(base_metadata)

        try:
            # Calcular relevância aprimorada
            enhanced_metadata["enhanced_relevance"] = (
                self._calculate_enhanced_relevance(text)
            )

            # Gerar embedding se o modelo estiver disponível
            if self.use_embeddings:
                embedding = self._generate_embedding(text)
                if embedding is not None:
                    enhanced_metadata["embedding"] = embedding
                    enhanced_metadata["embedding_model"] = self.embedding_model

            # Gerar resumo se o texto for suficientemente longo
            if len(text.split()) >= 30:
                if self.summary_algorithm == "textrank":
                    enhanced_metadata["summary"] = self._generate_textrank_summary(text)
                else:
                    enhanced_metadata["summary"] = self._generate_enhanced_summary(text)

            # Adicionar timestamp de análise
            enhanced_metadata["analyzed_at"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Erro ao aprimorar metadados: {e}")
            # Garantir que temos enhanced_relevance mesmo em caso de erro
            if "enhanced_relevance" not in enhanced_metadata:
                enhanced_metadata["enhanced_relevance"] = base_metadata.get(
                    "relevance_score", 0.0
                )

        return enhanced_metadata

    def _extract_base_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extrai metadados básicos do texto usando o analisador base.

        Args:
            text: Texto para análise

        Returns:
            Dicionário com metadados básicos
        """
        return super()._generate_metadata(text)

    def _calculate_similarities_in_streaming(
        self, results: List[AnalysisResult], similarity_matrix: Dict[str, np.ndarray]
    ) -> None:
        """
        Calcula similaridades entre chunks em modo streaming para economia de memória.

        Args:
            results: Lista de resultados de análise
            similarity_matrix: Dicionário de embeddings por ID de chunk
        """
        logger.info(
            f"Calculando similaridade em streaming entre {len(similarity_matrix)} chunks"
        )

        chunk_ids = list(similarity_matrix.keys())

        # Processa em pequenos lotes para economia de memória
        batch_size = min(50, len(chunk_ids))

        for i, result in enumerate(results):
            chunk_id = result.chunk_id

            if chunk_id not in similarity_matrix:
                continue

            # Calcular similaridade com outros chunks em lotes
            similarity_scores = {}
            embedding1 = similarity_matrix[chunk_id]

            for j in range(0, len(chunk_ids), batch_size):
                batch_ids = chunk_ids[j : j + batch_size]

                for other_id in batch_ids:
                    if other_id != chunk_id:
                        try:
                            embedding2 = similarity_matrix[other_id]
                            similarity = self._cosine_similarity(embedding1, embedding2)
                            similarity_scores[other_id] = float(similarity)
                        except Exception as e:
                            logger.error(f"Erro ao calcular similaridade: {str(e)}")

            # Adicionar scores de similaridade aos metadados
            if similarity_scores:
                result.metadata["similarity_scores"] = similarity_scores

            # Log a cada 100 resultados
            if i % 100 == 0 and i > 0:
                logger.info(
                    f"Calculadas similaridades para {i}/{len(results)} resultados"
                )

                # Liberar memória periodicamente
                if self.memory_efficient:
                    import gc

                    gc.collect()

    def _calculate_similarities_within_batch(
        self, batch_results: List[AnalysisResult]
    ) -> None:
        """
        Calcula similaridade entre os chunks dentro de um mesmo lote.

        Args:
            batch_results: Lista de resultados de análise do lote.
        """
        # Coletar embeddings
        embeddings_map = {}
        for result in batch_results:
            if "embedding" in result.metadata:
                embeddings_map[result.chunk_id] = np.array(result.metadata["embedding"])

        # Verificar se temos embeddings suficientes
        if len(embeddings_map) < 2:
            return

        # Calcular similaridade para cada resultado
        for result in batch_results:
            if result.chunk_id in embeddings_map:
                similarity_scores = result.metadata.get("similarity_scores", {})
                embedding1 = embeddings_map[result.chunk_id]

                for chunk_id, embedding2 in embeddings_map.items():
                    if chunk_id != result.chunk_id:
                        try:
                            similarity = self._cosine_similarity(embedding1, embedding2)
                            similarity_scores[chunk_id] = float(similarity)
                        except Exception as e:
                            logger.error(f"Erro ao calcular similaridade: {str(e)}")

                result.metadata["similarity_scores"] = similarity_scores

    def _log_memory_usage(self, context: str) -> None:
        """
        Registra o uso atual de memória.

        Args:
            context: Descrição do contexto atual para o log
        """
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            logger.info(
                f"Uso de memória {context}: {memory_info.rss / (1024 * 1024):.2f} MB"
            )

            # Se disponível, tentar obter info de VRAM (para GPU)
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                    logger.info(
                        f"Uso de VRAM {context}: {gpu_memory:.2f} MB (alocado) / {gpu_memory_reserved:.2f} MB (reservado)"
                    )
            except ImportError:
                pass

        except Exception as e:
            logger.warning(f"Não foi possível medir uso de memória: {e}")

    def clear_memory(self) -> None:
        """
        Libera memória após processamento.
        """
        try:
            # Limpar cache de modelos
            if self.sentence_transformer and hasattr(
                self.sentence_transformer, "clear_cache"
            ):
                self.sentence_transformer.clear_cache()

            # Coletor de lixo do Python
            import gc

            gc.collect()

            # Se PyTorch estiver disponível, tentar limpar cache CUDA
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cache CUDA liberado")
            except ImportError:
                pass

            logger.info("Memória liberada com sucesso")

        except Exception as e:
            logger.warning(f"Erro ao liberar memória: {e}")

    def _process_batch(
        self, task: AnalysisTask, chunks: List[ContentChunk]
    ) -> List[AnalysisResult]:
        """
        Processa um lote de chunks com verificação de tamanho para evitar problemas de memória.

        Args:
            task: Tarefa de análise.
            chunks: Lote de chunks para análise.

        Returns:
            Lista de resultados para o lote.
        """
        results = []

        # Verificar se algum chunk excede o tamanho máximo
        large_chunks = [
            chunk for chunk in chunks if len(chunk.text) > self.max_text_length
        ]
        regular_chunks = [
            chunk for chunk in chunks if len(chunk.text) <= self.max_text_length
        ]

        # Processar chunks grandes individualmente em streaming
        for chunk in large_chunks:
            logger.info(
                f"Processando chunk grande ({len(chunk.text)/1024:.2f} KB) individualmente"
            )
            result = self._process_large_chunk_in_streaming(chunk)
            results.append(result)

        # Processar chunks de tamanho regular em lote
        if regular_chunks:
            # Pré-carregar embeddings em lote se estiver usando embeddings
            embeddings = {}
            if self.use_embeddings and self.sentence_transformer:
                texts = [chunk.text for chunk in regular_chunks if chunk.text.strip()]
                if texts:
                    try:
                        batch_embeddings = self.sentence_transformer.encode(
                            texts, show_progress_bar=False
                        )
                        for i, chunk in enumerate(regular_chunks):
                            if chunk.text.strip():
                                embeddings[chunk.id] = batch_embeddings[i]
                    except Exception as e:
                        logger.error(f"Erro ao gerar embeddings em lote: {str(e)}")

            # Processar cada chunk do lote
            for chunk in regular_chunks:
                # Verificar se o ID do chunk está na lista de IDs da tarefa
                if chunk.id not in task.chunk_ids:
                    logger.warning(
                        f"Chunk {chunk.id} não está na lista de chunks da tarefa {task.id}"
                    )
                    continue

                # Gerar metadata básico com o analisador pai
                metadata = self._generate_base_metadata(chunk.text)

                # Adicionar relevância aprimorada
                metadata["relevance"] = self._calculate_enhanced_relevance(chunk.text)

                # Adicionar embedding se solicitado
                if self.use_embeddings:
                    if chunk.id in embeddings:
                        metadata["embedding"] = embeddings[chunk.id].tolist()
                        metadata["embedding_model"] = self.embedding_model
                    elif self.sentence_transformer and chunk.text.strip():
                        try:
                            embedding = self.sentence_transformer.encode(chunk.text)
                            metadata["embedding"] = embedding.tolist()
                            metadata["embedding_model"] = self.embedding_model
                        except Exception as e:
                            logger.error(
                                f"Erro ao gerar embedding para chunk {chunk.id}: {str(e)}"
                            )

                # Gerar resumo se solicitado
                if self.generate_summary and chunk.text.strip():
                    if self.summary_algorithm == "textrank":
                        metadata["summary"] = self._generate_textrank_summary(
                            chunk.text
                        )
                    else:
                        metadata["summary"] = self._generate_enhanced_summary(
                            chunk.text
                        )

                # Criar resultado
                result = AnalysisResult(
                    id=str(uuid.uuid4()),
                    task_id=task.id,
                    chunk_id=chunk.id,
                    metadata=metadata,
                    created_at=datetime.now(),
                )

                # Salvar resultado
                self._save_result(result)

                results.append(result)

        return results

    def _generate_base_metadata(self, text: str) -> Dict[str, Any]:
        """
        Gera metadados básicos usando o analisador pai.

        Args:
            text: Texto para análise.

        Returns:
            Dicionário com metadados.
        """
        # Chamar método de geração de metadados da classe pai
        metadata = super()._generate_metadata(text)

        # Renomear relevance_score para relevance para compatibilidade
        if "relevance_score" in metadata:
            metadata["relevance"] = metadata.pop("relevance_score")

        return metadata

    def _calculate_similarities(self, results: List[AnalysisResult]) -> None:
        """
        Calcula similaridade entre os chunks com base nos embeddings.

        Args:
            results: Lista de resultados de análise.
        """
        logger.info("Calculando similaridade entre chunks...")

        # Coletar embeddings
        embeddings_map = {}
        for result in results:
            if "embedding" in result.metadata:
                embeddings_map[result.chunk_id] = np.array(result.metadata["embedding"])

        # Verificar se temos embeddings suficientes
        if len(embeddings_map) < 2:
            logger.warning("Não há embeddings suficientes para calcular similaridade")
            return

        # Calcular similaridade para cada resultado
        for result in results:
            if result.chunk_id in embeddings_map:
                similarity_scores = {}
                embedding1 = embeddings_map[result.chunk_id]

                for chunk_id, embedding2 in embeddings_map.items():
                    if chunk_id != result.chunk_id:
                        try:
                            similarity = self._cosine_similarity(embedding1, embedding2)
                            similarity_scores[chunk_id] = float(similarity)
                        except Exception as e:
                            logger.error(f"Erro ao calcular similaridade: {str(e)}")

                result.metadata["similarity_scores"] = similarity_scores

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calcula similaridade de cosseno entre dois vetores.

        Args:
            vec1: Primeiro vetor.
            vec2: Segundo vetor.

        Returns:
            Similaridade de cosseno (entre 0 e 1).
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        dot_product = np.dot(vec1, vec2)
        similarity = dot_product / (norm1 * norm2)

        return float(similarity)

    def _generate_enhanced_summary(self, text: str) -> str:
        """
        Gera um resumo do texto usando TF-IDF para identificar sentenças importantes.

        Args:
            text: Texto para resumir.

        Returns:
            Resumo do texto.
        """
        # Verificar se o texto é longo o suficiente para resumo
        if not text.strip() or len(text.split()) < 30:
            return ""

        try:
            # Dividir o texto em sentenças
            from nltk.tokenize import sent_tokenize

            sentences = sent_tokenize(text)

            if len(sentences) < 3:
                return text if len(sentences) == 1 else " ".join(sentences)

            # Calcular TF-IDF para as sentenças
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Calcular importância de cada sentença como a soma dos valores TF-IDF
            sentence_scores = [
                sum(tfidf_matrix[i].toarray()[0]) for i in range(len(sentences))
            ]

            # Selecionar top N sentenças (30% do total, pelo menos 3, no máximo 5)
            num_sentences = max(3, min(5, int(len(sentences) * 0.3)))
            top_indices = sorted(
                range(len(sentence_scores)),
                key=lambda i: sentence_scores[i],
                reverse=True,
            )[:num_sentences]

            # Ordenar os índices para manter a ordem original do texto
            top_indices = sorted(top_indices)

            # Construir resumo
            summary = " ".join([sentences[i] for i in top_indices])

            return summary

        except Exception as e:
            logger.error(f"Erro ao gerar resumo: {str(e)}")
            # Retornar as primeiras sentenças como fallback
            from nltk.tokenize import sent_tokenize

            sentences = sent_tokenize(text)
            return " ".join(sentences[:3]) if len(sentences) >= 3 else text

    def _generate_textrank_summary(self, text: str) -> str:
        """
        Gera um resumo usando o algoritmo TextRank baseado em grafos.

        Args:
            text: Texto para resumir.

        Returns:
            Resumo do texto.
        """
        # Verificar se o texto é longo o suficiente para resumo
        if not text.strip() or len(text.split()) < 30:
            return ""

        try:
            # Dividir o texto em sentenças
            from nltk.tokenize import sent_tokenize

            sentences = sent_tokenize(text)

            if len(sentences) < 3:
                return text if len(sentences) == 1 else " ".join(sentences)

            # Criar matriz de similaridade entre sentenças
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

            # Criar grafo
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph, max_iter=100)

            # Selecionar top N sentenças (30% do total, pelo menos 3, no máximo 5)
            num_sentences = max(3, min(5, int(len(sentences) * 0.3)))
            ranked_sentences = sorted(
                ((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True
            )
            top_sentences = [
                ranked_sentences[i][2]
                for i in range(min(num_sentences, len(ranked_sentences)))
            ]

            # Ordenar as sentenças na ordem original do texto
            ordered_top_sentences = []
            for sentence in sentences:
                if sentence in top_sentences:
                    ordered_top_sentences.append(sentence)

            # Construir resumo
            summary = " ".join(ordered_top_sentences)

            return summary

        except Exception as e:
            logger.error(f"Erro ao gerar resumo com TextRank: {str(e)}")
            # Fallback para o método TF-IDF
            return self._generate_enhanced_summary(text)

    def _save_result(self, result: AnalysisResult) -> None:
        """
        Salva o resultado em arquivo se o diretório de armazenamento estiver definido.

        Args:
            result: Resultado de análise.
        """
        if self.storage_dir:
            try:
                result_path = os.path.join(self.storage_dir, f"{result.id}.json")
                with open(result_path, "w") as f:
                    f.write(result.model_dump_json(indent=2))
            except Exception as e:
                logger.error(f"Erro ao salvar resultado {result.id}: {str(e)}")

    def _load_embedding_model(self):
        """
        Carrega o modelo de embeddings sob demanda.
        Usa o gerenciador de memória para otimizar uso de RAM/VRAM.
        """
        if self.sentence_transformer is not None:
            return

        try:
            logger.info(f"Carregando modelo de embeddings: {self.embedding_model}")
            from sentence_transformers import SentenceTransformer

            # Usar o gerenciador de memória para carregar o modelo
            self.sentence_transformer = memory_manager.load_model(
                model_name=f"embedding_{self.embedding_model}",
                model_class=SentenceTransformer,
                model_name_or_path=self.embedding_model,
                device=self.device,
            )

            logger.info(
                f"Modelo de embeddings carregado com sucesso: {self.embedding_model}"
            )
        except ImportError:
            logger.warning(
                "Biblioteca sentence-transformers não encontrada. "
                "Embeddings avançados não estarão disponíveis."
            )
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de embeddings: {e}")
            self.sentence_transformer = None

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Gera um embedding vetorial para o texto fornecido.
        Utiliza o cache do gerenciador de memória para otimizar o desempenho.

        Args:
            text: Texto para gerar embedding

        Returns:
            Lista de valores do embedding ou None em caso de erro
        """
        if not text or len(text.strip()) < 10:
            return None

        # Gerar um ID único para o texto (usando um hash)
        import hashlib

        text_id = hashlib.md5(text.encode("utf-8")).hexdigest()

        # Verificar se existe no cache do gerenciador de memória
        cached_embedding = memory_manager.get_cached_embedding(text_id)
        if cached_embedding is not None:
            return cached_embedding

        try:
            # Carregar modelo se necessário
            if self.sentence_transformer is None:
                self._load_embedding_model()

            if self.sentence_transformer is None:
                return None

            # Gerar embedding
            embedding = self.sentence_transformer.encode(text, convert_to_tensor=False)

            # Armazenar no cache do gerenciador de memória
            memory_manager.cache_embedding(text_id, embedding)

            return embedding
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {e}")
            return None
