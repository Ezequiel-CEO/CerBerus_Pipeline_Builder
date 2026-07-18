#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analisador de conteúdo para o PipelineBuilder.

Avalia relevância, extrai tópicos e categoriza o texto.
"""

import os
import re
import json
import string
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
import logging
from collections import Counter
import uuid

# import spacy  # Removido para simplificar dependências
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# from sklearn.feature_extraction.text import TfidfVectorizer  # Removido temporariamente
# from gensim.summarization import keywords  # Removido temporariamente

from cerberus_api.pipeline_builder.models import (
    ContentChunk,
    AnalysisResult,
    ContentCategory,
    AnalysisTask,
    TaskStatus,
)
from cerberus_api.utils.logging_config import APILogger

# Configurar logger
logger = APILogger("content_analyzer")

# Baixar recursos do NLTK se necessário
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    logger.info("Baixando recursos do NLTK...")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

# Removida inicialização do spaCy
nlp = None

# Lista de idiomas suportados para análise avançada
SUPPORTED_LANGUAGES = {
    "en": "english",
    "pt": "portuguese",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "nl": "dutch",
    "ru": "russian",
    "ar": "arabic",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
}

# Mapeamento de códigos ISO para nomes de idiomas
LANGUAGE_NAMES = {
    "en": "English",
    "pt": "Portuguese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
    "ru": "Russian",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}

# Mapeamento de nomes de idiomas para códigos ISO
LANGUAGE_CODES = {v.lower(): k for k, v in LANGUAGE_NAMES.items()}


class ContentAnalyzer:
    """
    Analisador de conteúdo para avaliar texto e extrair informações relevantes.
    """

    def __init__(
        self,
        min_relevance_score: float = 0.5,
        keywords_ratio: float = 0.05,
        stopwords_langs: List[str] = ["portuguese", "english"],
        storage_dir: Optional[str] = None,
        extract_entities: bool = False,
        extract_sentiment: bool = False,
        nlp_model: str = "pt_core_news_sm",
    ):
        """
        Inicializa o analisador de conteúdo.

        Args:
            min_relevance_score: Pontuação mínima de relevância (0.0 a 1.0).
            keywords_ratio: Proporção de palavras para extrair como palavras-chave.
            stopwords_langs: Idiomas para stopwords.
            storage_dir: Diretório para armazenamento dos resultados de análise.
            extract_entities: Se deve extrair entidades nomeadas.
            extract_sentiment: Se deve analisar sentimento.
            nlp_model: Modelo spaCy a ser utilizado (se extract_entities ou extract_sentiment for True).
        """
        self.min_relevance_score = min_relevance_score
        self.keywords_ratio = keywords_ratio
        self.storage_dir = storage_dir
        self.extract_entities = extract_entities
        self.extract_sentiment = extract_sentiment
        self.nlp_model = nlp_model

        # Criar diretório de armazenamento se necessário
        if self.storage_dir:
            os.makedirs(self.storage_dir, exist_ok=True)
            logger.info(f"Diretório de armazenamento: {self.storage_dir}")

        # Inicializar recursos do NLTK
        self.stop_words = set()
        for lang in stopwords_langs:
            try:
                self.stop_words.update(stopwords.words(lang))
            except:
                logger.warning(f"Falha ao carregar stopwords para {lang}")

        self.lemmatizer = WordNetLemmatizer()

        # Inicializar spaCy se necessário
        self.nlp = None
        if self.extract_entities or self.extract_sentiment:
            try:
                import spacy

                self.nlp = spacy.load(self.nlp_model)
                logger.info(f"Modelo spaCy '{self.nlp_model}' carregado com sucesso")
            except Exception as e:
                logger.warning(f"Erro ao carregar modelo spaCy: {e}")
                logger.warning(
                    f"Para instalar o modelo, execute: python -m spacy download {self.nlp_model}"
                )

        # Padrões de categorização
        self.category_patterns = {
            ContentCategory.TECHNOLOGY: [
                "tecnologia",
                "software",
                "hardware",
                "computador",
                "digital",
                "rede",
                "internet",
                "programação",
                "algoritmo",
                "sistema",
                "banco de dados",
                "nuvem",
                "aplicativo",
                "desenvolvimento",
                "web",
                "móvel",
                "app",
                "processor",
                "memória",
                "código",
                "programa",
                "engenharia",
                "tech",
                "computação",
                "servidor",
                "aplicação",
                "interface",
                "API",
                "framework",
            ],
            ContentCategory.SCIENCE: [
                "ciência",
                "pesquisa",
                "científico",
                "experimento",
                "teoria",
                "estudo",
                "análise",
                "dados",
                "hipótese",
                "laboratório",
                "metodologia",
                "discovery",
                "física",
                "química",
                "biologia",
                "matemática",
                "estatística",
                "research",
                "neurociência",
                "genética",
                "investigação",
                "fenômeno",
                "medição",
            ],
            ContentCategory.SECURITY: [
                "segurança",
                "proteção",
                "criptografia",
                "privacidade",
                "firewall",
                "hacker",
                "vulnerabilidade",
                "ameaça",
                "risco",
                "ataque",
                "defesa",
                "malware",
                "autenticação",
                "acesso",
                "senha",
                "vírus",
                "cibersegurança",
                "exploração",
                "breach",
                "invasão",
                "pentest",
                "informação",
                "dado",
                "confidencial",
            ],
            ContentCategory.PROGRAMMING: [
                "programação",
                "código",
                "desenvolvedor",
                "software",
                "engenharia",
                "linguagem",
                "debug",
                "bug",
                "compilador",
                "interpretador",
                "framework",
                "biblioteca",
                "API",
                "função",
                "método",
                "classe",
                "objeto",
                "variável",
                "estrutura de dados",
                "algoritmo",
                "desenvolvimento",
                "teste",
                "script",
                "git",
                "github",
                "versão",
                "python",
                "java",
                "javascript",
                "typescript",
                "C++",
                "C#",
                "rust",
                "go",
            ],
            ContentCategory.TUTORIALS: [
                "tutorial",
                "guia",
                "passo a passo",
                "como fazer",
                "instrução",
                "aprenda",
                "ensinando",
                "demonstração",
                "exemplo",
                "prática",
                "exercício",
                "aula",
                "lição",
                "introdução",
                "começando",
                "básico",
                "iniciante",
                "curso",
                "howto",
                "aprendizado",
                "implementação",
                "configuração",
                "setup",
            ],
            ContentCategory.DOCUMENTATION: [
                "documentação",
                "manual",
                "referência",
                "especificação",
                "API",
                "guia",
                "documento",
                "instruções",
                "definição",
                "descrição",
                "changelog",
                "versão",
                "release",
                "resource",
                "biblioteca",
                "framework",
                "package",
                "módulo",
                "função",
                "classe",
                "método",
                "parâmetro",
                "retorno",
                "exemplo",
            ],
            ContentCategory.REFERENCE: [
                "referência",
                "fonte",
                "citação",
                "bibliografia",
                "autor",
                "publicação",
                "trabalho",
                "artigo",
                "paper",
                "livro",
                "jornal",
                "revista",
                "conferência",
                "simpósio",
                "workshop",
                "seminário",
                "estudo",
                "pesquisa",
                "experimento",
                "análise",
                "revisão",
                "survey",
                "literatura",
            ],
            ContentCategory.ACADEMIC: [
                "acadêmico",
                "universidade",
                "faculdade",
                "instituto",
                "educação",
                "pesquisa",
                "estudo",
                "tese",
                "dissertação",
                "professor",
                "aluno",
                "estudante",
                "curso",
                "graduação",
                "pós-graduação",
                "mestrado",
                "doutorado",
                "científico",
                "artigo",
                "publicação",
                "journal",
                "academia",
                "escolar",
                "pedagógico",
                "didático",
                "ensino",
                "aprendizagem",
            ],
        }

    def analyze(self, chunk: ContentChunk) -> AnalysisResult:
        """
        Analisa um chunk de conteúdo, extraindo metadados relevantes.

        Args:
            chunk: Chunk de conteúdo para análise

        Returns:
            Resultado da análise com metadados
        """
        try:
            # Gerar metadados a partir do texto do chunk
            metadata = self._generate_metadata(chunk.text)

            # Verificar se é relevante
            if metadata["relevance_score"] < self.min_relevance_score:
                # Se não for relevante, retornar resultado com relevância baixa
                return self._create_low_relevance_result(chunk.id)

            # Criar resultado
            result = AnalysisResult(
                id=str(uuid.uuid4()),
                task_id=chunk.task_id,  # Usar o task_id do chunk
                chunk_id=chunk.id,
                metadata=metadata,
                created_at=datetime.now(),
            )

            # Salvar se necessário
            if self.storage_dir:
                result_file = os.path.join(self.storage_dir, f"{result.id}.json")
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write(result.model_dump_json(indent=2))

            return result

        except Exception as e:
            logger.error(f"Erro ao analisar chunk {chunk.id}: {e}")

            # Retornar resultado de erro
            return AnalysisResult(
                id=str(uuid.uuid4()),
                task_id=chunk.task_id,  # Usar o task_id do chunk
                chunk_id=chunk.id,
                metadata={"error": str(e), "analyzed_at": datetime.now().isoformat()},
                created_at=datetime.now(),
            )

    def _calculate_relevance(self, text: str) -> float:
        """
        Calcula a pontuação de relevância do conteúdo.

        Args:
            text: Texto para análise.

        Returns:
            Pontuação de relevância (0.0 a 1.0).
        """
        # Implementação básica - avalia com base em:
        # 1. Comprimento e densidade do texto
        # 2. Coesão e estrutura
        # 3. Presença de termos técnicos
        # 4. Qualidade da informação

        score = 0.0

        # 1. Comprimento e densidade do texto
        words = word_tokenize(text.lower())
        if not words:
            return 0.0

        # Remover stopwords e normalizar
        filtered_words = [
            w for w in words if w not in self.stop_words and w not in string.punctuation
        ]

        # Calcular densidade de informação
        info_density = len(filtered_words) / max(len(words), 1)

        # Comprimento do texto
        length_score = min(
            1.0, len(filtered_words) / 1000
        )  # Normalizar para 1000 palavras

        # 2. Coesão e estrutura
        sentences = sent_tokenize(text)
        avg_sentence_length = len(words) / max(len(sentences), 1)

        # Penalizar sentenças muito curtas ou muito longas
        sentence_score = 1.0
        if avg_sentence_length < 5:
            sentence_score = 0.7
        elif avg_sentence_length > 40:
            sentence_score = 0.8

        # 3. Presença de termos técnicos
        # Simplificado para não depender da biblioteca spaCy
        tech_terms = []
        for category in self.category_patterns.values():
            tech_terms.extend(category)

        tech_term_count = sum(
            1
            for word in filtered_words
            if word.lower() in tech_terms
            or any(term in word.lower() for term in tech_terms if len(term) > 5)
        )

        tech_score = min(1.0, tech_term_count / (len(filtered_words) * 0.1))

        # 4. Calcular pontuação final
        score = (
            (length_score * 0.3)
            + (info_density * 0.3)
            + (sentence_score * 0.2)
            + (tech_score * 0.2)
        )

        return min(1.0, max(0.0, score))

    def _categorize_content(self, text: str) -> List[ContentCategory]:
        """
        Categoriza o conteúdo com base em padrões pré-definidos.

        Args:
            text: Texto para análise.

        Returns:
            Lista de categorias identificadas.
        """
        # Normalizar texto para análise
        text_lower = text.lower()

        # Calcular score para cada categoria
        scores = {}

        for category, patterns in self.category_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in text_lower:
                    score += text_lower.count(pattern)

            scores[category] = score

        # Filtra categorias com pontuação acima de um threshold
        threshold = max(1, len(text.split()) / 500)  # Adaptativo ao tamanho do texto

        # Retorna as categorias identificadas
        categories = [cat for cat, score in scores.items() if score > threshold]

        # Se nenhuma categoria foi encontrada, retorna a mais próxima
        if not categories and scores:
            top_category = max(scores.items(), key=lambda x: x[1])
            if top_category[1] > 0:
                categories = [top_category[0]]

        return categories

    def _extract_topics(self, text: str) -> List[str]:
        """
        Extrai tópicos/palavras-chave do texto.

        Args:
            text: Texto para análise.

        Returns:
            Lista de tópicos extraídos.
        """
        # Versão simplificada que não usa bibliotecas externas
        words = word_tokenize(text.lower())

        # Remover stopwords e pontuação
        filtered_words = [
            w
            for w in words
            if w not in self.stop_words and w not in string.punctuation and len(w) > 3
        ]

        # Contar frequência
        word_counts = Counter(filtered_words)

        # Extrair os N tópicos mais frequentes
        num_keywords = max(5, int(len(filtered_words) * self.keywords_ratio))
        topics = [word for word, count in word_counts.most_common(num_keywords)]

        return topics

    def _extract_entities_simple(self, text: str) -> Dict[str, List[str]]:
        """
        Extrai entidades do texto usando um método simples.

        Args:
            text: Texto para análise.

        Returns:
            Dicionário com entidades por tipo.
        """
        # Versão simplificada sem spaCy
        entities = {
            "ORG": [],  # Organizações
            "PERSON": [],  # Pessoas
            "TECH": [],  # Termos técnicos
        }

        # Procurar organizações comuns
        org_patterns = [
            r"\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)+\b(?:\s+Inc\.?|\s+Corp\.?|\s+LLC|\s+Ltd\.?)?",  # Empresas
            r"\b[A-Z]{2,}(?:\.[A-Z]{2,})*\b",  # Siglas
        ]

        for pattern in org_patterns:
            for match in re.finditer(pattern, text):
                org = match.group(0)
                if org not in entities["ORG"] and len(org) > 2:
                    entities["ORG"].append(org)

        # Identificar termos técnicos a partir dos padrões de categorização
        tech_terms = []
        for category in [ContentCategory.TECHNOLOGY, ContentCategory.PROGRAMMING]:
            tech_terms.extend(self.category_patterns.get(category, []))

        for term in tech_terms:
            if term in text.lower() and term not in entities["TECH"]:
                entities["TECH"].append(term)

        return entities

    # Função auxiliar para converter o dicionário de entidades para a lista de entidades
    def _convert_entities_to_list(
        self, entities_dict: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Converte um dicionário de entidades para uma lista no formato esperado pelo AnalysisResult.

        Args:
            entities_dict: Dicionário de entidades por tipo.

        Returns:
            Lista de entidades no formato esperado.
        """
        result = []
        for entity_type, entities in entities_dict.items():
            for entity in entities:
                result.append({"text": entity, "label": entity_type})
        return result

    def _generate_summary(self, text: str) -> str:
        """
        Gera um resumo do texto.

        Args:
            text: Texto para resumir.

        Returns:
            Resumo do texto.
        """
        # Versão simplificada que seleciona as sentenças mais importantes
        sentences = sent_tokenize(text)

        if not sentences:
            return ""

        if len(sentences) <= 3:
            return text

        # Extrair palavras-chave
        keywords = self._extract_topics(text)

        # Pontuação para cada sentença
        sentence_scores = {}

        for i, sentence in enumerate(sentences):
            score = 0

            # Score maior para primeiras e últimas sentenças
            if i == 0 or i == len(sentences) - 1:
                score += 2

            # Score para sentenças que contêm palavras-chave
            words = word_tokenize(sentence.lower())
            for word in words:
                if word in keywords:
                    score += 1

            # Normalizar score pelo comprimento
            score = score / max(1, len(words))

            sentence_scores[i] = score

        # Selecionar as sentenças com maior score
        num_summary_sentences = max(1, min(5, len(sentences) // 5))
        top_sentences = sorted(
            sentence_scores.items(), key=lambda x: x[1], reverse=True
        )[:num_summary_sentences]
        top_sentences = sorted(
            top_sentences, key=lambda x: x[0]
        )  # Ordenar por posição original

        summary = " ".join(sentences[i] for i, _ in top_sentences)

        return summary

    def _create_low_relevance_result(self, chunk_id: str) -> AnalysisResult:
        """
        Cria um resultado para conteúdo com baixa relevância.

        Args:
            chunk_id: ID do chunk

        Returns:
            Resultado da análise com metadados mínimos
        """
        return AnalysisResult(
            id=str(uuid.uuid4()),
            task_id="default_task",  # Task ID padrão para resultados de baixa relevância
            chunk_id=chunk_id,
            metadata={
                "relevance_score": 0.0,
                "low_relevance": True,
                "analyzed_at": datetime.now().isoformat(),
                "categories": [ContentCategory.OTHER.value],
            },
            created_at=datetime.now(),
        )

    def _detect_language(self, text: str) -> Tuple[str, str]:
        """
        Detecta o idioma do texto.

        Args:
            text: Texto para detecção de idioma.

        Returns:
            Tupla com (código_idioma, nome_idioma).
        """
        if not text.strip():
            return "unknown", "Unknown"

        try:
            # Usando detector baseado em fastText (comentado)
            # if self.language_detector:
            #     predictions = self.language_detector.predict(text.replace('\n', ' '))
            #     lang_code = predictions[0][0].replace('__label__', '')
            #     confidence = predictions[1][0]
            #
            #     if confidence > 0.5:
            #         return lang_code, LANGUAGE_NAMES.get(lang_code, 'Unknown')

            # Abordagem heurística simples baseada em contagem de palavras comuns
            text = text.lower()

            # Definir palavras comuns para alguns idiomas principais
            common_words = {
                "pt": [
                    "de",
                    "o",
                    "a",
                    "que",
                    "e",
                    "do",
                    "da",
                    "em",
                    "um",
                    "para",
                    "com",
                    "não",
                    "uma",
                    "os",
                ],
                "en": [
                    "the",
                    "of",
                    "and",
                    "a",
                    "to",
                    "in",
                    "is",
                    "you",
                    "that",
                    "it",
                    "he",
                    "was",
                    "for",
                    "on",
                ],
                "es": [
                    "de",
                    "la",
                    "que",
                    "el",
                    "en",
                    "y",
                    "a",
                    "los",
                    "del",
                    "se",
                    "las",
                    "por",
                    "un",
                    "para",
                ],
                "fr": [
                    "de",
                    "la",
                    "le",
                    "et",
                    "les",
                    "des",
                    "en",
                    "un",
                    "du",
                    "une",
                    "que",
                    "est",
                    "pour",
                    "qui",
                ],
                "de": [
                    "der",
                    "die",
                    "und",
                    "den",
                    "von",
                    "zu",
                    "das",
                    "mit",
                    "sich",
                    "des",
                    "auf",
                    "für",
                    "ist",
                    "im",
                ],
                "it": [
                    "di",
                    "e",
                    "il",
                    "la",
                    "che",
                    "in",
                    "a",
                    "per",
                    "un",
                    "con",
                    "su",
                    "da",
                    "del",
                    "si",
                ],
            }

            # Tokenizar texto
            words = set(word.strip(string.punctuation) for word in text.split())

            # Contar ocorrências de palavras comuns para cada idioma
            lang_scores = {}
            for lang, common_list in common_words.items():
                common_count = sum(1 for word in words if word in common_list)
                lang_scores[lang] = common_count / len(common_list)

            if lang_scores:
                # Obter idioma com maior pontuação
                detected_lang = max(lang_scores.items(), key=lambda x: x[1])

                # Verificar se a pontuação é significativa
                if detected_lang[1] > 0.2:
                    lang_code = detected_lang[0]
                    return lang_code, LANGUAGE_NAMES.get(lang_code, "Unknown")

            # Fallback: língua portuguesa
            return "pt", "Portuguese"

        except Exception as e:
            logger.error(f"Erro ao detectar idioma: {e}")
            return "pt", "Portuguese"  # Fallback para português

    def _get_stop_words(self, lang_code: str) -> Set[str]:
        """
        Obtém conjunto de stop words para o idioma especificado.

        Args:
            lang_code: Código ISO do idioma.

        Returns:
            Conjunto de stop words.
        """
        try:
            # Usar NLTK para obter stop words
            if nltk and lang_code in SUPPORTED_LANGUAGES:
                lang_name = SUPPORTED_LANGUAGES[lang_code]
                return set(stopwords.words(lang_name))

            # Fallback: stop words básicas
            basic_stopwords = {
                "pt": [
                    "a",
                    "ao",
                    "aos",
                    "aquela",
                    "aquelas",
                    "aquele",
                    "aqueles",
                    "aquilo",
                    "as",
                    "até",
                    "com",
                    "como",
                    "da",
                    "das",
                    "de",
                    "dela",
                    "delas",
                    "dele",
                    "deles",
                    "depois",
                    "do",
                    "dos",
                    "e",
                    "ela",
                    "elas",
                    "ele",
                    "eles",
                    "em",
                    "entre",
                    "era",
                    "eram",
                    "essa",
                    "essas",
                    "esse",
                    "esses",
                    "esta",
                    "estas",
                    "este",
                    "estes",
                    "eu",
                    "foi",
                    "foram",
                    "há",
                    "isso",
                    "isto",
                    "já",
                    "lhe",
                    "lhes",
                    "mais",
                    "mas",
                    "me",
                    "mesmo",
                    "meu",
                    "meus",
                    "minha",
                    "minhas",
                    "muito",
                    "na",
                    "não",
                    "nas",
                    "nem",
                    "no",
                    "nos",
                    "nós",
                    "nossa",
                    "nossas",
                    "nosso",
                    "nossos",
                    "num",
                    "numa",
                    "o",
                    "os",
                    "ou",
                    "para",
                    "pela",
                    "pelas",
                    "pelo",
                    "pelos",
                    "por",
                    "qual",
                    "quando",
                    "que",
                    "quem",
                    "sao",
                    "se",
                    "seja",
                    "sem",
                    "seu",
                    "seus",
                    "só",
                    "somos",
                    "são",
                    "sua",
                    "suas",
                    "também",
                    "te",
                    "tem",
                    "tém",
                    "temos",
                    "ter",
                    "teu",
                    "teus",
                    "tu",
                    "tua",
                    "tuas",
                    "um",
                    "uma",
                    "você",
                    "vocês",
                    "vos",
                ],
                "en": [
                    "a",
                    "an",
                    "the",
                    "and",
                    "but",
                    "if",
                    "or",
                    "because",
                    "as",
                    "until",
                    "while",
                    "of",
                    "at",
                    "by",
                    "for",
                    "with",
                    "about",
                    "against",
                    "between",
                    "into",
                    "through",
                    "during",
                    "before",
                    "after",
                    "above",
                    "below",
                    "to",
                    "from",
                    "up",
                    "down",
                    "in",
                    "out",
                    "on",
                    "off",
                    "over",
                    "under",
                    "again",
                    "further",
                    "then",
                    "once",
                    "here",
                    "there",
                    "when",
                    "where",
                    "why",
                    "how",
                    "all",
                    "any",
                    "both",
                    "each",
                    "few",
                    "more",
                    "most",
                    "other",
                    "some",
                    "such",
                    "no",
                    "nor",
                    "not",
                    "only",
                    "own",
                    "same",
                    "so",
                    "than",
                    "too",
                    "very",
                    "can",
                    "will",
                    "just",
                    "don",
                    "should",
                    "now",
                ],
            }

            return set(basic_stopwords.get(lang_code, basic_stopwords["en"]))

        except Exception as e:
            logger.error(f"Erro ao obter stop words: {e}")
            return set()

    def _extract_sentiment(self, text: str) -> float:
        """
        Analisa o sentimento do texto.

        Args:
            text: Texto para análise

        Returns:
            Pontuação de sentimento entre -1.0 (negativo) e 1.0 (positivo)
        """
        # Análise simples baseada em palavras-chave para demonstração
        positive_words = [
            "bom",
            "excelente",
            "ótimo",
            "incrível",
            "fantástico",
            "maravilhoso",
            "útil",
            "good",
            "excellent",
            "great",
            "amazing",
            "fantastic",
            "wonderful",
            "helpful",
        ]

        negative_words = [
            "ruim",
            "terrível",
            "horrível",
            "péssimo",
            "inútil",
            "problemático",
            "bad",
            "terrible",
            "horrible",
            "awful",
            "useless",
            "problematic",
        ]

        words = word_tokenize(text.lower())

        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        # Se não há palavras de sentimento, retornar neutro
        if positive_count == 0 and negative_count == 0:
            return 0.0

        # Caso contrário, calcular pontuação
        total = positive_count + negative_count
        score = (positive_count - negative_count) / total

        return score

    def _generate_metadata(self, text: str) -> Dict[str, Any]:
        """
        Gera metadados a partir do texto.

        Args:
            text: Texto para análise

        Returns:
            Dicionário com metadados
        """
        # Verificar se o texto é válido
        if not text or not text.strip():
            return {
                "relevance_score": 0.0,
                "language": "unknown",
                "word_count": 0,
                "sentence_count": 0,
                "keywords": [],
                "entities": [],
                "topics": [],
                "categories": [ContentCategory.OTHER.value],
                "sentiment_score": 0.0,
            }

        # Detectar idioma
        lang_code, lang_name = self._detect_language(text)

        # Tokenizar o texto
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())

        # Contar palavras e sentenças
        word_count = len(words)
        sentence_count = len(sentences)

        # Calcular relevância
        relevance_score = self._calculate_relevance(text)

        # Extrair tópicos/palavras-chave
        keywords = self._extract_topics(text)

        # Categorizar conteúdo
        categories = [cat.value for cat in self._categorize_content(text)]

        # Extrair entidades nomeadas (versão simples)
        entity_dict = self._extract_entities_simple(text)
        entities = self._convert_entities_to_list(entity_dict)

        # Analisar sentimento
        sentiment_score = (
            self._extract_sentiment(text) if self.extract_sentiment else 0.0
        )

        # Construir metadados
        metadata = {
            "relevance_score": relevance_score,
            "language": lang_name,
            "language_code": lang_code,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "keywords": keywords,
            "entities": entities,
            "topics": keywords[
                : min(5, len(keywords))
            ],  # Tópicos são as palavras-chave mais relevantes
            "categories": categories,
            "sentiment_score": sentiment_score,
            "analyzed_at": datetime.now().isoformat(),
        }

        return metadata
