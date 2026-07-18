#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de feedback para aprendizado do Èter

Este módulo implementa mecanismos para o Èter receber e processar
feedback do usuário, permitindo aprendizado incremental e adaptação.
"""

import os
import json
import time
import datetime
import logging
from typing import Dict, List, Any, Optional, Union

from cerberus_api.utils.logging_config import get_logger


class EterFeedback:
    """
    Sistema de processamento de feedback para aprendizado do Èter.

    Gerencia o ciclo de feedback entre o usuário e o assistente,
    permitindo a melhoria contínua do desempenho e adaptação às
    necessidades específicas do usuário.
    """

    def __init__(
        self, storage_dir: str, user_id: str = "default_user", verbose: bool = False
    ):
        """
        Inicializa o sistema de feedback.

        Args:
            storage_dir: Diretório para armazenamento dos feedbacks
            user_id: ID do usuário atual
            verbose: Se True, exibe logs detalhados
        """
        self.storage_dir = os.path.abspath(storage_dir)
        self.user_id = user_id
        self.verbose = verbose

        # Configuração de logging
        self.logger = get_logger(f"eter_feedback_{user_id}")

        # Garantir que o diretório de armazenamento exista
        self.feedback_dir = os.path.join(self.storage_dir, "feedback")
        os.makedirs(self.feedback_dir, exist_ok=True)

        # Estatísticas de feedback
        self.stats = {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "suggestions": 0,
            "total": 0,
        }

        self._load_stats()

        self.logger.info(f"Sistema de feedback inicializado para usuário: {user_id}")
        self.logger.debug(f"Diretório de feedback: {self.feedback_dir}")

    def _load_stats(self) -> None:
        """Carrega estatísticas de feedback do armazenamento."""
        stats_file = os.path.join(self.feedback_dir, "stats.json")

        if os.path.exists(stats_file):
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    self.stats = json.load(f)
                self.logger.debug(f"Estatísticas de feedback carregadas: {self.stats}")
            except Exception as e:
                self.logger.error(f"Erro ao carregar estatísticas: {e}")

    def _save_stats(self) -> None:
        """Salva estatísticas de feedback no armazenamento."""
        stats_file = os.path.join(self.feedback_dir, "stats.json")

        try:
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            self.logger.debug("Estatísticas de feedback salvas")
        except Exception as e:
            self.logger.error(f"Erro ao salvar estatísticas: {e}")

    def add_feedback(
        self,
        feedback_type: str,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        interaction_id: Optional[str] = None,
        rating: Optional[int] = None,
    ) -> str:
        """
        Registra um feedback do usuário.

        Args:
            feedback_type: Tipo de feedback (positive, negative, neutral, suggestion)
            content: Conteúdo do feedback
            context: Contexto em que o feedback foi dado
            interaction_id: ID da interação relacionada ao feedback
            rating: Avaliação numérica (1-5, se aplicável)

        Returns:
            ID do feedback registrado
        """
        # Validar o tipo de feedback
        if feedback_type not in ("positive", "negative", "neutral", "suggestion"):
            feedback_type = "neutral"
            self.logger.warning(f"Tipo de feedback inválido, usando 'neutral'")

        # Criar registro de feedback
        timestamp = datetime.datetime.now().isoformat()
        feedback_id = f"fb_{int(time.time())}_{feedback_type}"

        feedback = {
            "id": feedback_id,
            "timestamp": timestamp,
            "type": feedback_type,
            "user_id": self.user_id,
            "content": content,
            "context": context or {},
            "interaction_id": interaction_id,
            "rating": rating,
        }

        # Armazenar feedback
        filepath = os.path.join(self.feedback_dir, f"{feedback_id}.json")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(feedback, f, ensure_ascii=False, indent=2)

            # Atualizar estatísticas
            self.stats[feedback_type] += 1
            self.stats["total"] += 1
            self._save_stats()

            self.logger.info(f"Feedback registrado: {feedback_id} ({feedback_type})")
            self.logger.debug(
                f"Conteúdo: {content[:50]}{'...' if len(content) > 50 else ''}"
            )

            return feedback_id

        except Exception as e:
            self.logger.error(f"Erro ao armazenar feedback: {e}")
            return ""

    def get_feedback(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """
        Recupera um feedback específico pelo ID.

        Args:
            feedback_id: ID do feedback

        Returns:
            Dicionário com dados do feedback ou None se não encontrado
        """
        filepath = os.path.join(self.feedback_dir, f"{feedback_id}.json")

        if not os.path.exists(filepath):
            self.logger.warning(f"Feedback não encontrado: {feedback_id}")
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                feedback = json.load(f)

            self.logger.debug(f"Feedback recuperado: {feedback_id}")
            return feedback

        except Exception as e:
            self.logger.error(f"Erro ao ler feedback {feedback_id}: {e}")
            return None

    def get_feedback_by_type(
        self, feedback_type: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recupera feedbacks por tipo.

        Args:
            feedback_type: Tipo de feedback a filtrar
            limit: Número máximo de feedbacks a retornar

        Returns:
            Lista de feedbacks do tipo especificado
        """
        feedbacks = []

        try:
            files = os.listdir(self.feedback_dir)

            for filename in files:
                if not filename.endswith(".json") or filename == "stats.json":
                    continue

                filepath = os.path.join(self.feedback_dir, filename)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        feedback = json.load(f)

                    if feedback.get("type") == feedback_type:
                        feedbacks.append(feedback)

                        if len(feedbacks) >= limit:
                            break

                except Exception as e:
                    self.logger.warning(f"Erro ao ler arquivo {filename}: {e}")

            self.logger.info(
                f"Recuperados {len(feedbacks)} feedbacks do tipo {feedback_type}"
            )

            # Ordenar por timestamp (mais recentes primeiro)
            feedbacks.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            return feedbacks

        except Exception as e:
            self.logger.error(f"Erro ao buscar feedbacks: {e}")
            return []

    def get_feedback_stats(self) -> Dict[str, int]:
        """
        Retorna estatísticas de feedback.

        Returns:
            Dicionário com contagens por tipo de feedback
        """
        return self.stats

    def get_recent_feedback(
        self, days: int = 7, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Recupera feedbacks recentes.

        Args:
            days: Número de dias para considerar como recente
            limit: Número máximo de feedbacks a retornar

        Returns:
            Lista de feedbacks recentes
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        feedbacks = []

        try:
            files = os.listdir(self.feedback_dir)

            for filename in files:
                if not filename.endswith(".json") or filename == "stats.json":
                    continue

                filepath = os.path.join(self.feedback_dir, filename)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        feedback = json.load(f)

                    timestamp = feedback.get("timestamp", "")
                    if timestamp:
                        fb_date = datetime.datetime.fromisoformat(timestamp)
                        if fb_date >= cutoff_date:
                            feedbacks.append(feedback)

                except Exception as e:
                    self.logger.warning(f"Erro ao ler arquivo {filename}: {e}")

            self.logger.info(f"Recuperados {len(feedbacks)} feedbacks recentes")

            # Ordenar por timestamp (mais recentes primeiro)
            feedbacks.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            # Limitar quantidade
            return feedbacks[:limit]

        except Exception as e:
            self.logger.error(f"Erro ao buscar feedbacks recentes: {e}")
            return []

    def generate_feedback_summary(self) -> Dict[str, Any]:
        """
        Gera um resumo dos feedbacks recentes e tendências.

        Returns:
            Resumo de feedbacks
        """
        summary = {
            "total_count": self.stats["total"],
            "distribution": {
                "positive": self.stats["positive"],
                "negative": self.stats["negative"],
                "neutral": self.stats["neutral"],
                "suggestions": self.stats["suggestions"],
            },
            "positive_rate": 0,
            "recent_count": 0,
            "recent_trend": "stable",
        }

        # Calcular taxa de positividade
        if summary["total_count"] > 0:
            summary["positive_rate"] = round(
                (self.stats["positive"] / summary["total_count"]) * 100, 1
            )

        # Analisar tendência recente
        recent_feedbacks = self.get_recent_feedback(days=7)
        summary["recent_count"] = len(recent_feedbacks)

        # Calcular tendência
        recent_positive = sum(
            1 for fb in recent_feedbacks if fb.get("type") == "positive"
        )
        recent_negative = sum(
            1 for fb in recent_feedbacks if fb.get("type") == "negative"
        )

        if summary["recent_count"] > 0:
            recent_positive_rate = recent_positive / summary["recent_count"]

            if recent_positive_rate > 0.7:
                summary["recent_trend"] = "very_positive"
            elif recent_positive_rate > 0.5:
                summary["recent_trend"] = "positive"
            elif recent_positive_rate < 0.3:
                summary["recent_trend"] = "negative"
            else:
                summary["recent_trend"] = "stable"

        self.logger.info(f"Resumo de feedback gerado: {summary['recent_trend']}")
        return summary
