#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sistema de memória para a Èter.

Gerencia o armazenamento e recuperação do histórico de conversas,
conhecimento adquirido e contexto de execução.
"""

import os
import json
import time
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import pickle

from cerberus_api.utils.logging_config import APILogger

logger = APILogger("eter_memory")


class EterMemory:
    """
    Sistema de memória para a Èter.

    Gerencia:
    - Memória de curto prazo (sessão atual)
    - Memória de longo prazo (persistente)
    - Conhecimento técnico (documentação, etc.)
    - Embeddings para recuperação semântica
    """

    def __init__(
        self,
        storage_dir: str = "data/eter/memory",
        embedding_function=None,
        max_short_term: int = 100,
        max_contexts: int = 10,
        user_id: str = "default_user",
        verbose: bool = False,
    ):
        """
        Inicializa o sistema de memória.

        Args:
            storage_dir: Diretório para armazenamento persistente
            embedding_function: Função para geração de embeddings
            max_short_term: Tamanho máximo da memória de curto prazo
            max_contexts: Número máximo de contextos ativos
            user_id: ID do usuário atual
            verbose: Se True, exibe logs detalhados
        """
        self.storage_dir = os.path.abspath(storage_dir)
        self.embedding_function = embedding_function
        self.max_short_term = max_short_term
        self.max_contexts = max_contexts
        self.user_id = user_id
        self.verbose = verbose

        # Criar diretórios de armazenamento
        self.conversation_dir = os.path.join(self.storage_dir, "conversations")
        self.knowledge_dir = os.path.join(self.storage_dir, "knowledge")
        self.context_dir = os.path.join(self.storage_dir, "contexts")
        self.embedding_dir = os.path.join(self.storage_dir, "embeddings")

        for dir_path in [
            self.conversation_dir,
            self.knowledge_dir,
            self.context_dir,
            self.embedding_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

        # Inicializar estruturas de memória
        self.short_term_memory = []  # Memória da sessão atual
        self.active_contexts = {}  # Contextos ativos
        self.embedding_cache = {}  # Cache de embeddings

        # Carregar memória persistente
        self._load_memory()

        logger.info(f"Sistema de memória inicializado: {self.storage_dir}")

    def add_conversation(
        self,
        role: str,
        content: str,
        context_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Adiciona uma entrada à conversa atual.

        Args:
            role: Papel (user, assistant)
            content: Conteúdo da mensagem
            context_id: ID do contexto
            metadata: Metadados adicionais

        Returns:
            ID da entrada
        """
        timestamp = datetime.now().isoformat()
        entry_id = hashlib.md5(
            f"{role}_{timestamp}_{content[:50]}".encode()
        ).hexdigest()

        entry = {
            "id": entry_id,
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "context_id": context_id,
            "metadata": metadata or {},
        }

        # Adicionar à memória de curto prazo
        self.short_term_memory.append(entry)

        # Limitar tamanho da memória de curto prazo
        if len(self.short_term_memory) > self.max_short_term:
            self._consolidate_memory()

        # Salvar no armazenamento persistente
        try:
            conversation_file = os.path.join(
                self.conversation_dir,
                f"{context_id}_{datetime.now().strftime('%Y%m%d')}.jsonl",
            )

            with open(conversation_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.warning(f"Erro ao salvar entrada de conversa: {e}")

        return entry_id

    def get_conversation_history(
        self,
        context_id: str = "default",
        limit: int = 10,
        include_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Obtém histórico de conversa para um contexto.

        Args:
            context_id: ID do contexto
            limit: Número máximo de entradas
            include_metadata: Incluir metadados

        Returns:
            Lista de entradas de conversa
        """
        # Buscar na memória de curto prazo primeiro
        entries = [
            entry
            for entry in self.short_term_memory
            if entry["context_id"] == context_id
        ]

        # Se não houver entradas suficientes, buscar no armazenamento persistente
        if len(entries) < limit:
            try:
                # Buscar arquivos de conversa para este contexto
                files = [
                    f
                    for f in os.listdir(self.conversation_dir)
                    if f.startswith(f"{context_id}_") and f.endswith(".jsonl")
                ]

                # Ordenar por data (mais recentes primeiro)
                files.sort(reverse=True)

                # Ler entradas dos arquivos
                additional_entries = []
                remaining = limit - len(entries)

                for file in files:
                    if remaining <= 0:
                        break

                    file_path = os.path.join(self.conversation_dir, file)

                    with open(file_path, "r", encoding="utf-8") as f:
                        file_entries = [json.loads(line) for line in f]

                    # Ordenar por timestamp (mais recentes primeiro)
                    file_entries.sort(
                        key=lambda e: e.get("timestamp", ""), reverse=True
                    )

                    # Adicionar entradas
                    additional_entries.extend(file_entries[:remaining])
                    remaining -= len(file_entries[:remaining])

                # Mesclar com entradas de curto prazo
                entries = entries + additional_entries

            except Exception as e:
                logger.warning(f"Erro ao carregar histórico de conversa: {e}")

        # Ordenar por timestamp
        entries.sort(key=lambda e: e.get("timestamp", ""))

        # Limitar ao número solicitado
        entries = entries[-limit:]

        # Remover metadados se solicitado
        if not include_metadata:
            entries = [{k: v for k, v in e.items() if k != "metadata"} for e in entries]

        return entries

    def store_context(
        self, context_id: str, data: Dict[str, Any], persist: bool = True
    ) -> bool:
        """
        Armazena contexto ativo.

        Args:
            context_id: ID do contexto
            data: Dados do contexto
            persist: Persistir no armazenamento

        Returns:
            True se bem sucedido
        """
        # Adicionar timestamp
        data["_updated_at"] = datetime.now().isoformat()

        # Armazenar no dicionário de contextos ativos
        self.active_contexts[context_id] = data

        # Limitar número de contextos ativos
        if len(self.active_contexts) > self.max_contexts:
            # Remover contexto mais antigo
            oldest_key = min(
                self.active_contexts.keys(),
                key=lambda k: self.active_contexts[k].get("_updated_at", ""),
            )
            del self.active_contexts[oldest_key]

        # Persistir se solicitado
        if persist:
            try:
                context_file = os.path.join(self.context_dir, f"{context_id}.json")

                with open(context_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                return True

            except Exception as e:
                logger.warning(f"Erro ao persistir contexto: {e}")
                return False

        return True

    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtém contexto pelo ID.

        Args:
            context_id: ID do contexto

        Returns:
            Dados do contexto ou None se não existir
        """
        # Verificar cache de contextos ativos
        if context_id in self.active_contexts:
            return self.active_contexts[context_id]

        # Verificar armazenamento persistente
        try:
            context_file = os.path.join(self.context_dir, f"{context_id}.json")

            if os.path.exists(context_file):
                with open(context_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Adicionar ao cache
                self.active_contexts[context_id] = data
                return data

        except Exception as e:
            logger.warning(f"Erro ao carregar contexto: {e}")

        return None

    def store_knowledge(
        self,
        key: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        generate_embedding: bool = True,
    ) -> bool:
        """
        Armazena conhecimento.

        Args:
            key: Chave única para o conhecimento
            content: Conteúdo do conhecimento
            metadata: Metadados adicionais
            generate_embedding: Gerar embedding para busca semântica

        Returns:
            True se bem sucedido
        """
        try:
            # Preparar dados
            data = {
                "key": key,
                "content": content,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
            }

            # Salvar conhecimento
            knowledge_file = os.path.join(self.knowledge_dir, f"{key}.json")

            with open(knowledge_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Gerar embedding se solicitado
            if generate_embedding and self.embedding_function:
                embedding = self.embedding_function(content)

                # Salvar embedding
                embedding_file = os.path.join(self.embedding_dir, f"{key}.pkl")

                with open(embedding_file, "wb") as f:
                    pickle.dump(embedding, f)

                # Adicionar ao cache
                self.embedding_cache[key] = embedding

            return True

        except Exception as e:
            logger.warning(f"Erro ao armazenar conhecimento: {e}")
            return False

    def search_knowledge(
        self, query: str, limit: int = 5, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Busca conhecimento por similaridade.

        Args:
            query: Consulta para busca
            limit: Número máximo de resultados
            threshold: Limiar de similaridade

        Returns:
            Lista de resultados
        """
        if not self.embedding_function:
            logger.warning("Função de embedding não disponível")
            return []

        try:
            # Gerar embedding para a consulta
            query_embedding = self.embedding_function(query)

            # Calcular similaridade com todos os embeddings
            similarities = []

            # Carregar embeddings se cache estiver vazio
            if not self.embedding_cache:
                self._load_embeddings()

            # Calcular similaridades
            for key, embedding in self.embedding_cache.items():
                similarity = self._calculate_similarity(query_embedding, embedding)

                if similarity >= threshold:
                    similarities.append((key, similarity))

            # Ordenar por similaridade (maior primeiro)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Limitar número de resultados
            similarities = similarities[:limit]

            # Carregar conhecimento para as chaves
            results = []

            for key, similarity in similarities:
                knowledge = self._load_knowledge(key)

                if knowledge:
                    knowledge["similarity"] = similarity
                    results.append(knowledge)

            return results

        except Exception as e:
            logger.warning(f"Erro ao buscar conhecimento: {e}")
            return []

    def _calculate_similarity(self, embedding1, embedding2) -> float:
        """Calcula similaridade entre dois embeddings."""
        import numpy as np

        # Converter para numpy arrays se necessário
        if hasattr(embedding1, "numpy"):
            embedding1 = embedding1.numpy()
        if hasattr(embedding2, "numpy"):
            embedding2 = embedding2.numpy()

        # Normalizar embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        # Calcular similaridade de cosseno
        return float(np.dot(embedding1, embedding2))

    def _load_knowledge(self, key: str) -> Optional[Dict[str, Any]]:
        """Carrega conhecimento do armazenamento."""
        try:
            knowledge_file = os.path.join(self.knowledge_dir, f"{key}.json")

            if os.path.exists(knowledge_file):
                with open(knowledge_file, "r", encoding="utf-8") as f:
                    return json.load(f)

        except Exception as e:
            logger.warning(f"Erro ao carregar conhecimento {key}: {e}")

        return None

    def _load_embeddings(self) -> None:
        """Carrega embeddings do armazenamento."""
        try:
            embedding_files = [
                f for f in os.listdir(self.embedding_dir) if f.endswith(".pkl")
            ]

            for file in embedding_files:
                key = file[:-4]  # Remover extensão .pkl
                embedding_file = os.path.join(self.embedding_dir, file)

                with open(embedding_file, "rb") as f:
                    embedding = pickle.load(f)

                self.embedding_cache[key] = embedding

            logger.info(f"Carregados {len(self.embedding_cache)} embeddings")

        except Exception as e:
            logger.warning(f"Erro ao carregar embeddings: {e}")

    def _consolidate_memory(self) -> None:
        """
        Consolida memória de curto prazo em longo prazo.
        Move entradas mais antigas para armazenamento persistente.
        """
        # Manter apenas as últimas entradas
        if len(self.short_term_memory) <= self.max_short_term:
            return

        # Número de entradas a mover
        to_move = len(self.short_term_memory) - self.max_short_term

        # Separar entradas a serem movidas por contexto
        entries_by_context = {}

        for entry in self.short_term_memory[:to_move]:
            context_id = entry["context_id"]

            if context_id not in entries_by_context:
                entries_by_context[context_id] = []

            entries_by_context[context_id].append(entry)

        # Persistir entradas por contexto
        for context_id, entries in entries_by_context.items():
            try:
                conversation_file = os.path.join(
                    self.conversation_dir,
                    f"{context_id}_{datetime.now().strftime('%Y%m%d')}.jsonl",
                )

                with open(conversation_file, "a", encoding="utf-8") as f:
                    for entry in entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            except Exception as e:
                logger.warning(
                    f"Erro ao consolidar memória para contexto {context_id}: {e}"
                )

        # Atualizar memória de curto prazo
        self.short_term_memory = self.short_term_memory[to_move:]

        logger.info(
            f"Memória consolidada: {to_move} entradas movidas para armazenamento persistente"
        )

    def _load_memory(self) -> None:
        """Carrega dados iniciais de memória."""
        # Os dados serão carregados sob demanda
        logger.info("Sistema de memória pronto para uso")

    def clear_short_term(self) -> None:
        """Limpa memória de curto prazo."""
        self.short_term_memory = []

    def clear_all(self) -> None:
        """Limpa todas as estruturas de memória (não remove arquivos)."""
        self.short_term_memory = []
        self.active_contexts = {}
        self.embedding_cache = {}
