#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilitários de NLP para o PipelineBuilder.

Este módulo fornece funções para processamento de linguagem natural,
incluindo extração de entidades, análise de sentimento, e mais.
"""

import re
import string
import hashlib
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import Counter
import logging

# Tentar importar bibliotecas opcionais
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, RSLPStemmer
    from nltk.probability import FreqDist

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from cerberus_api.utils.logging_config import APILogger

# Configurar logger
logger = APILogger("nlp_utils")

# Garantir que recursos do NLTK estejam disponíveis
if NLTK_AVAILABLE:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        logger.info("Baixando recursos do NLTK...")
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

# Carregar modelos spaCy se disponíveis
nlp_pt = None
nlp_en = None

if SPACY_AVAILABLE:
    try:
        nlp_pt = spacy.load("pt_core_news_sm")
        logger.info("Modelo spaCy para português carregado.")
    except (OSError, IOError):
        logger.warning(
            "Modelo spaCy para português não encontrado. Execute: python -m spacy download pt_core_news_sm"
        )

    try:
        nlp_en = spacy.load("en_core_web_sm")
        logger.info("Modelo spaCy para inglês carregado.")
    except (OSError, IOError):
        logger.warning(
            "Modelo spaCy para inglês não encontrado. Execute: python -m spacy download en_core_web_sm"
        )


def preprocess_text(text: str, lang: str = "pt") -> str:
    """
    Pré-processa o texto para análise.

    Args:
        text: Texto para pré-processar.
        lang: Código do idioma ('pt' ou 'en').

    Returns:
        Texto pré-processado.
    """
    if not text:
        return ""

    # Converter para minúsculas
    text = text.lower()

    # Remover URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remover e-mails
    text = re.sub(r"\S+@\S+", "", text)

    # Remover HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remover símbolos especiais e pontuação
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # Remover números
    text = re.sub(r"\d+", "", text)

    # Remover espaços extras
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_keywords(text: str, n: int = 10, lang: str = "pt") -> List[str]:
    """
    Extrai as principais palavras-chave de um texto.

    Args:
        text: Texto para análise.
        n: Número de palavras-chave para extrair.
        lang: Código do idioma ('pt' ou 'en').

    Returns:
        Lista de palavras-chave.
    """
    if not text or not NLTK_AVAILABLE:
        return []

    # Pré-processar texto
    processed_text = preprocess_text(text, lang)

    # Tokenizar
    tokens = word_tokenize(processed_text)

    # Remover stopwords
    try:
        stop_words = set(stopwords.words("portuguese" if lang == "pt" else "english"))
    except:
        stop_words = set(
            [
                "o",
                "a",
                "os",
                "as",
                "um",
                "uma",
                "uns",
                "umas",
                "de",
                "do",
                "da",
                "dos",
                "das",
                "the",
                "a",
                "an",
                "of",
                "in",
                "on",
                "at",
                "by",
                "for",
                "with",
                "about",
                "to",
            ]
        )

    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # Lematizar
    if lang == "pt":
        try:
            stemmer = RSLPStemmer()
            tokens = [stemmer.stem(t) for t in tokens]
        except Exception as e:
            logger.warning(f"Erro ao aplicar stemming em português: {e}")
    else:
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
        except Exception as e:
            logger.warning(f"Erro ao aplicar lemmatização em inglês: {e}")

    # Calcular frequência
    fdist = FreqDist(tokens)

    # Retornar as n palavras mais frequentes
    return [item[0] for item in fdist.most_common(n)]


def extract_named_entities(text: str, lang: str = "pt") -> List[Dict[str, Any]]:
    """
    Extrai entidades nomeadas de um texto.

    Args:
        text: Texto para análise.
        lang: Código do idioma ('pt' ou 'en').

    Returns:
        Lista de entidades encontradas.
    """
    entities = []

    # Usar spaCy se disponível
    if SPACY_AVAILABLE:
        nlp = nlp_pt if lang == "pt" else nlp_en

        if nlp:
            try:
                # Limitar tamanho para evitar problemas de memória
                max_length = min(len(text), 100000)
                doc = nlp(text[:max_length])

                for ent in doc.ents:
                    entity = {
                        "text": ent.text,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "label": ent.label_,
                    }
                    entities.append(entity)

                return entities
            except Exception as e:
                logger.warning(f"Erro na extração de entidades com spaCy: {e}")

    # Fallback para abordagem baseada em regex
    # Extrair possíveis nomes próprios (palavras capitalizadas)
    if not entities:
        try:
            # Tokenizar o texto em frases
            sentences = sent_tokenize(text)

            for sent in sentences:
                # Encontrar palavras capitalizadas que não estão no início da frase
                words = sent.split()

                for i, word in enumerate(words):
                    # Ignorar a primeira palavra da frase
                    if i == 0:
                        continue

                    # Verificar se a palavra começa com maiúscula
                    if word and word[0].isupper():
                        # Verificar palavras adjacentes para entidades multi-palavra
                        start = i
                        end = i

                        # Olhar para trás
                        while start > 0 and words[start - 1][0].isupper():
                            start -= 1

                        # Olhar para frente
                        while end < len(words) - 1 and words[end + 1][0].isupper():
                            end += 1

                        # Extrair a entidade
                        entity_text = " ".join(words[start : end + 1])

                        # Limpar pontuação no final
                        entity_text = re.sub(r"[^\w\s]$", "", entity_text)

                        if entity_text and len(entity_text) > 2:
                            entity = {"text": entity_text, "label": "UNKNOWN"}

                            # Adicionar se ainda não estiver na lista
                            if not any(e["text"] == entity_text for e in entities):
                                entities.append(entity)

            return entities

        except Exception as e:
            logger.warning(f"Erro na extração de entidades com regex: {e}")

    return entities


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calcula a similaridade entre dois textos usando TF-IDF e similaridade de cosseno.

    Args:
        text1: Primeiro texto.
        text2: Segundo texto.

    Returns:
        Valor de similaridade entre 0 e 1.
    """
    if not text1 or not text2:
        return 0.0

    if SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Erro ao calcular similaridade com TF-IDF: {e}")

    # Fallback para cálculo de similaridade baseado em palavras comuns
    try:
        # Pré-processar textos
        processed_text1 = set(preprocess_text(text1).split())
        processed_text2 = set(preprocess_text(text2).split())

        # Calcular similaridade de Jaccard
        intersection = len(processed_text1.intersection(processed_text2))
        union = len(processed_text1.union(processed_text2))

        if union == 0:
            return 0.0

        return intersection / union

    except Exception as e:
        logger.warning(f"Erro ao calcular similaridade básica: {e}")
        return 0.0


def generate_text_summary(text: str, ratio: float = 0.2) -> str:
    """
    Gera um resumo extrativo do texto.

    Args:
        text: Texto para resumir.
        ratio: Proporção do texto original a manter no resumo (0-1).

    Returns:
        Resumo do texto.
    """
    if not text:
        return ""

    # Se o texto for pequeno, retornar original
    if len(text.split()) < 100:
        return text

    try:
        # Tokenizar em frases
        sentences = sent_tokenize(text)

        if len(sentences) <= 3:
            return text

        # Pré-processar cada frase
        processed_sentences = [preprocess_text(sent) for sent in sentences]

        # Calcular score para cada frase
        sentence_scores = {}

        # Score baseado na posição
        for i, _ in enumerate(sentences):
            # Frases no início e fim tendem a ser mais importantes
            position_score = (
                1.0 if i < len(sentences) * 0.2 or i > len(sentences) * 0.8 else 0.5
            )
            sentence_scores[i] = position_score

        # Score baseado em palavras-chave
        if NLTK_AVAILABLE:
            # Extrair palavras-chave do texto completo
            keywords = extract_keywords(text, n=20)

            for i, sent in enumerate(processed_sentences):
                # Contar ocorrências de palavras-chave
                keyword_count = sum(1 for word in sent.split() if word in keywords)
                keyword_score = min(keyword_count / 5, 1.0)  # Normalizar para máx 1.0

                # Adicionar ao score existente
                sentence_scores[i] = sentence_scores.get(i, 0) + keyword_score

        # Score baseado em tamanho da frase
        for i, sent in enumerate(sentences):
            words = len(sent.split())
            # Frases médias geralmente são mais informativas (não muito curtas nem muito longas)
            length_score = 0.5
            if 5 <= words <= 20:
                length_score = 1.0
            elif words < 5:
                length_score = 0.3

            # Adicionar ao score existente
            sentence_scores[i] = sentence_scores.get(i, 0) + length_score * 0.5

        # Determinar número de frases a manter
        num_sentences = max(3, int(len(sentences) * ratio))

        # Selecionar as melhores frases
        best_sentences = sorted(
            range(len(sentences)), key=lambda i: sentence_scores.get(i, 0), reverse=True
        )[:num_sentences]

        # Ordenar pelo índice original para manter a ordem do texto
        best_sentences = sorted(best_sentences)

        # Gerar resumo
        summary = " ".join([sentences[i] for i in best_sentences])

        return summary

    except Exception as e:
        logger.warning(f"Erro ao gerar resumo: {e}")
        return text[: int(len(text) * ratio)]  # Fallback para truncamento simples


def detect_language(text: str) -> str:
    """
    Detecta o idioma do texto.

    Args:
        text: Texto para análise.

    Returns:
        Código do idioma ('pt', 'en', ou 'unknown').
    """
    if not text:
        return "unknown"

    if SPACY_AVAILABLE:
        try:
            # Usar spaCy para detecção de idioma
            # Pegar uma amostra do texto para análise rápida
            sample = text[:1000]

            # Tentar com modelo português primeiro
            if nlp_pt:
                doc_pt = nlp_pt(sample)
                pt_score = sum(
                    1 for t in doc_pt if not t.is_punct and not t.is_space
                ) / len(doc_pt)
            else:
                pt_score = 0

            # Tentar com modelo inglês
            if nlp_en:
                doc_en = nlp_en(sample)
                en_score = sum(
                    1 for t in doc_en if not t.is_punct and not t.is_space
                ) / len(doc_en)
            else:
                en_score = 0

            # Determinar idioma baseado no melhor score
            if pt_score > en_score:
                return "pt"
            elif en_score > 0:
                return "en"
            else:
                return "unknown"

        except Exception as e:
            logger.warning(f"Erro na detecção de idioma com spaCy: {e}")

    # Fallback para abordagem baseada em stopwords
    if NLTK_AVAILABLE:
        try:
            # Pré-processar e tokenizar
            processed_text = preprocess_text(text)
            words = word_tokenize(processed_text)

            # Amostrar apenas 100 palavras para eficiência
            words = words[:100]

            # Carregar stopwords para cada idioma
            pt_stops = set(stopwords.words("portuguese"))
            en_stops = set(stopwords.words("english"))

            # Contar ocorrências de stopwords
            pt_count = sum(1 for word in words if word in pt_stops)
            en_count = sum(1 for word in words if word in en_stops)

            # Determinar idioma baseado no maior número de stopwords
            if pt_count > en_count:
                return "pt"
            elif en_count > pt_count:
                return "en"
            else:
                return "unknown"

        except Exception as e:
            logger.warning(f"Erro na detecção de idioma com NLTK: {e}")

    # Abordagem simples baseada em caracteres específicos
    # Português tem mais acentos e caracteres especiais que inglês
    pt_chars = set("áàãâéêíóôõúüçÁÀÃÂÉÊÍÓÔÕÚÜÇ")
    text_chars = set(text)

    if len(text_chars.intersection(pt_chars)) > 0:
        return "pt"

    return "unknown"


def hash_text(text: str) -> str:
    """
    Gera um hash único para um texto.

    Args:
        text: Texto para hash.

    Returns:
        String hash do texto.
    """
    if not text:
        return ""

    # Remover espaços extras e normalizar quebras de linha
    normalized_text = re.sub(r"\s+", " ", text).strip()

    # Calcular hash MD5
    return hashlib.md5(normalized_text.encode("utf-8")).hexdigest()


def split_into_chunks(
    text: str, max_chunk_size: int = 5000, overlap: int = 200
) -> List[str]:
    """
    Divide um texto em chunks de tamanho aproximado.

    Args:
        text: Texto para dividir.
        max_chunk_size: Tamanho máximo de cada chunk (caracteres).
        overlap: Sobreposição entre chunks (caracteres).

    Returns:
        Lista de chunks de texto.
    """
    if not text:
        return []

    # Se o texto for menor que o tamanho máximo, retornar sem dividir
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []

    # Tentar dividir por parágrafos primeiro
    paragraphs = text.split("\n\n")

    current_chunk = ""

    for para in paragraphs:
        # Se o parágrafo for muito grande, dividir por frases
        if len(para) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # Dividir parágrafo grande em frases
            if NLTK_AVAILABLE:
                sentences = sent_tokenize(para)
            else:
                # Fallback para regex
                sentences = re.split(r"(?<=[.!?])\s+", para)

            sentence_chunk = ""

            for sentence in sentences:
                # Se a frase for muito grande, dividir arbitrariamente
                if len(sentence) > max_chunk_size:
                    if sentence_chunk:
                        chunks.append(sentence_chunk)
                        sentence_chunk = ""

                    # Dividir frase grande em pedaços
                    for i in range(0, len(sentence), max_chunk_size - overlap):
                        chunk_end = min(i + max_chunk_size, len(sentence))
                        chunks.append(sentence[i:chunk_end])

                        # Se chegarmos ao final, não precisamos adicionar mais
                        if chunk_end == len(sentence):
                            break

                elif len(sentence_chunk) + len(sentence) + 1 > max_chunk_size:
                    chunks.append(sentence_chunk)
                    sentence_chunk = sentence
                else:
                    if sentence_chunk:
                        sentence_chunk += " " + sentence
                    else:
                        sentence_chunk = sentence

            if sentence_chunk:
                chunks.append(sentence_chunk)

        # Se adicionar o parágrafo exceder o tamanho máximo, criar novo chunk
        elif len(current_chunk) + len(para) + 2 > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    # Adicionar o último chunk se não estiver vazio
    if current_chunk:
        chunks.append(current_chunk)

    # Garantir sobreposição se solicitado
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i in range(len(chunks)):
            if i == 0:
                overlapped_chunks.append(chunks[i])
            else:
                # Pegar o final do chunk anterior
                prev_end = (
                    chunks[i - 1][-overlap:]
                    if len(chunks[i - 1]) > overlap
                    else chunks[i - 1]
                )
                overlapped_chunks.append(prev_end + chunks[i])

        chunks = overlapped_chunks

    return chunks


def estimate_tokens(text_or_chunks, model_name="gpt-3.5"):
    """
    Estima o número de tokens em um texto para modelagem de linguagem.

    Args:
        text_or_chunks: Texto ou lista de tuplas (ContentChunk, AnalysisResult).
        model_name: Nome do modelo para ajustes específicos.

    Returns:
        Número estimado de tokens.
    """
    # Verificar se estamos recebendo uma lista de tuplas (chunk, result)
    if isinstance(text_or_chunks, list):
        # Se for uma lista, somar os tokens de cada chunk
        total_tokens = 0
        for chunk_tuple in text_or_chunks:
            if isinstance(chunk_tuple, tuple) and len(chunk_tuple) >= 1:
                chunk = chunk_tuple[0]
                chunk_text = chunk.text if hasattr(chunk, "text") else ""
                total_tokens += estimate_tokens(chunk_text, model_name)
        return total_tokens

    # Processar um único texto
    text = text_or_chunks
    if not text:
        return 0

    # Estimativa simplificada baseada em palavras e caracteres
    # Na média, 1 token ≈ 4 caracteres ou 0.75 palavras para inglês/português

    # Método 1: baseado em caracteres
    char_estimate = len(text) / 4

    # Método 2: baseado em palavras
    words = text.split()
    word_estimate = (
        len(words) * 1.3
    )  # Fator de ajuste para considerar tokens de pontuação

    # Usar a média entre os dois métodos
    estimate = (char_estimate + word_estimate) / 2

    # Ajustes específicos para certos modelos
    if "gpt-4" in model_name:
        # GPT-4 pode ser ligeiramente mais eficiente na tokenização
        estimate = estimate * 0.95

    return max(1, int(estimate))
