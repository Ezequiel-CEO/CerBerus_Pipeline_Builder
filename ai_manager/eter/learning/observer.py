#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de observação para aprendizado do Èter

Este módulo implementa mecanismos para o Èter observar e aprender
a partir de ações realizadas no ambiente CerBerus FMK.
"""

import os
import json
import time
import datetime
import logging
from typing import Dict, List, Any, Optional, Callable

from cerberus_api.utils.logging_config import get_logger


class EterObserver:
    """
    Sistema de observação para aprendizado incremental.

    Monitora ações realizadas no ambiente CerBerus, captura
    contexto e resultados, e armazena informações para
    aprendizado futuro.
    """

    def __init__(
        self, storage_dir: str, user_id: str = "default_user", verbose: bool = False
    ):
        """
        Inicializa o observador.

        Args:
            storage_dir: Diretório para armazenamento das observações
            user_id: ID do usuário atual
            verbose: Se True, exibe logs detalhados
        """
        self.storage_dir = os.path.abspath(storage_dir)
        self.user_id = user_id
        self.verbose = verbose

        # Configuração de logging
        self.logger = get_logger(f"eter_observer_{user_id}")

        # Garantir que o diretório de armazenamento exista
        self.observations_dir = os.path.join(self.storage_dir, "observations")
        os.makedirs(self.observations_dir, exist_ok=True)

        # Armazenamento de callbacks para notificação
        self._callbacks = []

        self.logger.info(f"Observador inicializado para usuário: {user_id}")
        self.logger.debug(f"Diretório de observações: {self.observations_dir}")

    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Registra um callback para ser notificado sobre novas observações.

        Args:
            callback: Função a ser chamada quando uma nova observação for registrada
        """
        self._callbacks.append(callback)
        self.logger.debug(f"Novo callback registrado. Total: {len(self._callbacks)}")

    def observe_action(
        self,
        action_type: str,
        context: Dict[str, Any],
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Registra uma observação de ação e seu resultado.

        Args:
            action_type: Tipo da ação observada
            context: Contexto da ação
            result: Resultado da ação
            metadata: Metadados adicionais

        Returns:
            ID da observação registrada
        """
        # Criar registro de observação
        timestamp = datetime.datetime.now().isoformat()
        observation_id = f"obs_{int(time.time())}_{action_type}"

        observation = {
            "id": observation_id,
            "timestamp": timestamp,
            "action_type": action_type,
            "user_id": self.user_id,
            "context": context,
            "result": result,
            "metadata": metadata or {},
        }

        # Armazenar observação
        filepath = os.path.join(self.observations_dir, f"{observation_id}.json")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(observation, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Observação registrada: {observation_id}")
            self.logger.debug(
                f"Detalhes: tipo={action_type}, resultado={result.get('status', 'unknown')}"
            )

            # Notificar callbacks
            for callback in self._callbacks:
                try:
                    callback(observation)
                except Exception as e:
                    self.logger.error(
                        f"Erro ao executar callback para observação {observation_id}: {e}"
                    )

            return observation_id

        except Exception as e:
            self.logger.error(f"Erro ao armazenar observação: {e}")
            return ""

    def get_observation(self, observation_id: str) -> Optional[Dict[str, Any]]:
        """
        Recupera uma observação específica pelo ID.

        Args:
            observation_id: ID da observação

        Returns:
            Dicionário com dados da observação ou None se não encontrada
        """
        filepath = os.path.join(self.observations_dir, f"{observation_id}.json")

        if not os.path.exists(filepath):
            self.logger.warning(f"Observação não encontrada: {observation_id}")
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                observation = json.load(f)

            self.logger.debug(f"Observação recuperada: {observation_id}")
            return observation

        except Exception as e:
            self.logger.error(f"Erro ao ler observação {observation_id}: {e}")
            return None

    def get_observations_by_action_type(
        self, action_type: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recupera observações por tipo de ação.

        Args:
            action_type: Tipo de ação a filtrar
            limit: Número máximo de observações a retornar

        Returns:
            Lista de observações do tipo especificado
        """
        observations = []

        try:
            files = os.listdir(self.observations_dir)

            for filename in files:
                if not filename.endswith(".json"):
                    continue

                filepath = os.path.join(self.observations_dir, filename)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        observation = json.load(f)

                    if observation.get("action_type") == action_type:
                        observations.append(observation)

                        if len(observations) >= limit:
                            break

                except Exception as e:
                    self.logger.warning(f"Erro ao ler arquivo {filename}: {e}")

            self.logger.info(
                f"Recuperadas {len(observations)} observações do tipo {action_type}"
            )

            # Ordenar por timestamp (mais recentes primeiro)
            observations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            return observations

        except Exception as e:
            self.logger.error(f"Erro ao buscar observações: {e}")
            return []

    def cleanup_old_observations(self, days_to_keep: int = 30) -> int:
        """
        Remove observações antigas do armazenamento.

        Args:
            days_to_keep: Número de dias para manter observações

        Returns:
            Número de observações removidas
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
        removed_count = 0

        try:
            files = os.listdir(self.observations_dir)

            for filename in files:
                if not filename.endswith(".json"):
                    continue

                filepath = os.path.join(self.observations_dir, filename)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        observation = json.load(f)

                    timestamp = observation.get("timestamp", "")
                    if timestamp:
                        obs_date = datetime.datetime.fromisoformat(timestamp)
                        if obs_date < cutoff_date:
                            os.remove(filepath)
                            removed_count += 1

                except Exception as e:
                    self.logger.warning(f"Erro ao processar arquivo {filename}: {e}")

            self.logger.info(
                f"Limpeza concluída: {removed_count} observações removidas"
            )
            return removed_count

        except Exception as e:
            self.logger.error(f"Erro durante limpeza de observações: {e}")
            return 0
