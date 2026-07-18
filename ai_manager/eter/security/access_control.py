#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Controle de acesso para o Èter

Este módulo implementa funcionalidades de controle de acesso para o Èter,
garantindo a segurança das operações realizadas pelo assistente.
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

from cerberus_api.utils.logging_config import get_logger


class AccessControl:
    """
    Sistema de controle de acesso para o Èter.

    Gerencia permissões, validação de operações sensíveis e
    registro de atividades para garantir a segurança do ambiente.
    """

    # Níveis de permissão definidos
    PERMISSION_LEVELS = {
        "read": 10,  # Somente leitura
        "execute": 20,  # Execução de tarefas existentes
        "write": 30,  # Modificação de recursos
        "admin": 40,  # Acesso administrativo completo
    }

    # Operações por nível de permissão
    OPERATIONS = {
        "read": {"query", "read_file", "list_dir", "get_status", "search"},
        "execute": {"run_pipeline", "analyze_text", "process_data", "export_results"},
        "write": {
            "create_pipeline",
            "modify_pipeline",
            "save_data",
            "create_file",
            "delete_file",
        },
        "admin": {
            "manage_users",
            "configure_system",
            "install_components",
            "update_system",
        },
    }

    def __init__(
        self, config_dir: str, user_id: str = "default_user", verbose: bool = False
    ):
        """
        Inicializa o sistema de controle de acesso.

        Args:
            config_dir: Diretório para configurações de segurança
            user_id: ID do usuário atual
            verbose: Se True, exibe logs detalhados
        """
        self.config_dir = os.path.abspath(config_dir)
        self.user_id = user_id
        self.verbose = verbose

        # Configuração de logging
        self.logger = get_logger(f"eter_access_{user_id}")

        # Garantir que o diretório de configuração exista
        self.security_dir = os.path.join(self.config_dir, "security")
        os.makedirs(self.security_dir, exist_ok=True)

        # Carregar configurações
        self.users_file = os.path.join(self.security_dir, "users.json")
        self.audit_log_file = os.path.join(self.security_dir, "audit.log")

        self.users = self._load_users()
        if not self.users:
            self._create_default_user()

        self.logger.info(f"Controle de acesso inicializado para usuário: {user_id}")

    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        """
        Carrega informações de usuários do arquivo de configuração.

        Returns:
            Dicionário com informações de usuários
        """
        if not os.path.exists(self.users_file):
            return {}

        try:
            with open(self.users_file, "r", encoding="utf-8") as f:
                users = json.load(f)

            self.logger.debug(
                f"Informações de usuários carregadas: {len(users)} usuários"
            )
            return users

        except Exception as e:
            self.logger.error(f"Erro ao carregar informações de usuários: {e}")
            return {}

    def _save_users(self) -> bool:
        """
        Salva informações de usuários no arquivo de configuração.

        Returns:
            True se a operação foi bem-sucedida, False caso contrário
        """
        try:
            with open(self.users_file, "w", encoding="utf-8") as f:
                json.dump(self.users, f, ensure_ascii=False, indent=2)

            self.logger.debug("Informações de usuários salvas com sucesso")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao salvar informações de usuários: {e}")
            return False

    def _create_default_user(self) -> None:
        """Cria um usuário padrão se nenhum usuário existir."""
        default_user = "default_user"
        self.users = {
            default_user: {
                "user_id": default_user,
                "permission_level": "admin",
                "api_key": self._generate_api_key(default_user),
                "created_at": time.time(),
                "last_access": time.time(),
                "enabled": True,
            }
        }

        self._save_users()
        self.logger.info(f"Usuário padrão criado: {default_user}")

    def _generate_api_key(self, user_id: str) -> str:
        """
        Gera uma chave de API para um usuário.

        Args:
            user_id: ID do usuário

        Returns:
            Chave de API gerada
        """
        seed = f"{user_id}:{time.time()}:{os.urandom(8).hex()}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    def _log_action(self, action: str, details: Dict[str, Any], success: bool) -> None:
        """
        Registra uma ação no log de auditoria.

        Args:
            action: Nome da ação executada
            details: Detalhes da ação
            success: Se a ação foi bem-sucedida
        """
        log_entry = {
            "timestamp": time.time(),
            "user_id": self.user_id,
            "action": action,
            "success": success,
            "details": details,
        }

        try:
            with open(self.audit_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            self.logger.error(f"Erro ao registrar ação no log de auditoria: {e}")

    def authenticate(self, user_id: str, api_key: str) -> bool:
        """
        Autentica um usuário usando ID e chave de API.

        Args:
            user_id: ID do usuário
            api_key: Chave de API do usuário

        Returns:
            True se a autenticação for bem-sucedida, False caso contrário
        """
        if user_id not in self.users:
            self.logger.warning(
                f"Tentativa de autenticação para usuário inexistente: {user_id}"
            )
            self._log_action("authenticate", {"user_id": user_id}, False)
            return False

        user = self.users[user_id]

        if not user.get("enabled", False):
            self.logger.warning(
                f"Tentativa de autenticação para usuário desativado: {user_id}"
            )
            self._log_action("authenticate", {"user_id": user_id}, False)
            return False

        if user.get("api_key") != api_key:
            self.logger.warning(f"Chave de API inválida para usuário: {user_id}")
            self._log_action("authenticate", {"user_id": user_id}, False)
            return False

        # Atualizar último acesso
        user["last_access"] = time.time()
        self._save_users()

        self.logger.info(f"Usuário autenticado com sucesso: {user_id}")
        self._log_action("authenticate", {"user_id": user_id}, True)

        return True

    def has_permission(self, operation: str) -> bool:
        """
        Verifica se o usuário atual tem permissão para executar uma operação.

        Args:
            operation: Nome da operação a ser verificada

        Returns:
            True se o usuário tem permissão, False caso contrário
        """
        if self.user_id not in self.users:
            self.logger.warning(f"Usuário não encontrado: {self.user_id}")
            return False

        user = self.users[self.user_id]
        user_level = user.get("permission_level", "read")

        # Mapear operação ao nível de permissão necessário
        required_level = None
        for level, operations in self.OPERATIONS.items():
            if operation in operations:
                required_level = level
                break

        if not required_level:
            self.logger.warning(f"Operação desconhecida: {operation}")
            return False

        # Verificar se o nível do usuário é suficiente
        user_level_value = self.PERMISSION_LEVELS.get(user_level, 0)
        required_level_value = self.PERMISSION_LEVELS.get(required_level, 0)

        has_permission = user_level_value >= required_level_value

        if not has_permission:
            self.logger.warning(
                f"Usuário {self.user_id} ({user_level}) não tem permissão "
                f"para a operação {operation} (requer {required_level})"
            )
            self._log_action(
                "permission_check",
                {"operation": operation, "required_level": required_level},
                False,
            )

        return has_permission

    def create_user(
        self, new_user_id: str, permission_level: str = "read"
    ) -> Tuple[bool, Optional[str]]:
        """
        Cria um novo usuário.

        Args:
            new_user_id: ID do novo usuário
            permission_level: Nível de permissão

        Returns:
            Tupla (sucesso, api_key)
        """
        # Verificar se o usuário atual tem permissão para criar usuários
        if not self.has_permission("manage_users"):
            self.logger.warning(
                f"Usuário {self.user_id} tentou criar um novo usuário sem permissão"
            )
            return False, None

        # Validar nível de permissão
        if permission_level not in self.PERMISSION_LEVELS:
            self.logger.warning(f"Nível de permissão inválido: {permission_level}")
            return False, None

        # Verificar se o usuário já existe
        if new_user_id in self.users:
            self.logger.warning(f"Usuário já existe: {new_user_id}")
            return False, None

        # Gerar chave de API
        api_key = self._generate_api_key(new_user_id)

        # Criar novo usuário
        self.users[new_user_id] = {
            "user_id": new_user_id,
            "permission_level": permission_level,
            "api_key": api_key,
            "created_at": time.time(),
            "created_by": self.user_id,
            "last_access": time.time(),
            "enabled": True,
        }

        success = self._save_users()

        if success:
            self.logger.info(f"Novo usuário criado: {new_user_id} ({permission_level})")
            self._log_action(
                "create_user",
                {"new_user_id": new_user_id, "permission_level": permission_level},
                True,
            )
            return True, api_key
        else:
            return False, None

    def modify_user(
        self,
        target_user_id: str,
        permission_level: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> bool:
        """
        Modifica um usuário existente.

        Args:
            target_user_id: ID do usuário a ser modificado
            permission_level: Novo nível de permissão
            enabled: Se o usuário está ativado

        Returns:
            True se a operação foi bem-sucedida, False caso contrário
        """
        # Verificar se o usuário atual tem permissão
        if not self.has_permission("manage_users"):
            self.logger.warning(
                f"Usuário {self.user_id} tentou modificar usuário sem permissão"
            )
            return False

        # Verificar se o usuário alvo existe
        if target_user_id not in self.users:
            self.logger.warning(f"Usuário alvo não existe: {target_user_id}")
            return False

        user = self.users[target_user_id]
        changes = {}

        # Atualizar nível de permissão
        if permission_level is not None:
            if permission_level not in self.PERMISSION_LEVELS:
                self.logger.warning(f"Nível de permissão inválido: {permission_level}")
                return False

            user["permission_level"] = permission_level
            changes["permission_level"] = permission_level

        # Atualizar status de ativação
        if enabled is not None:
            user["enabled"] = enabled
            changes["enabled"] = enabled

        user["last_modified"] = time.time()
        user["modified_by"] = self.user_id

        success = self._save_users()

        if success:
            self.logger.info(
                f"Usuário modificado: {target_user_id} (alterações: {changes})"
            )
            self._log_action(
                "modify_user",
                {"target_user_id": target_user_id, "changes": changes},
                True,
            )

        return success

    def reset_api_key(self, target_user_id: str) -> Tuple[bool, Optional[str]]:
        """
        Redefine a chave de API de um usuário.

        Args:
            target_user_id: ID do usuário

        Returns:
            Tupla (sucesso, nova_chave_api)
        """
        # Verificar se o usuário atual tem permissão
        if target_user_id != self.user_id and not self.has_permission("manage_users"):
            self.logger.warning(
                f"Usuário {self.user_id} tentou redefinir chave de API sem permissão"
            )
            return False, None

        # Verificar se o usuário alvo existe
        if target_user_id not in self.users:
            self.logger.warning(f"Usuário alvo não existe: {target_user_id}")
            return False, None

        # Gerar nova chave de API
        new_api_key = self._generate_api_key(target_user_id)

        # Atualizar informações do usuário
        user = self.users[target_user_id]
        user["api_key"] = new_api_key
        user["key_reset_at"] = time.time()
        user["key_reset_by"] = self.user_id

        success = self._save_users()

        if success:
            self.logger.info(f"Chave de API redefinida para usuário: {target_user_id}")
            self._log_action("reset_api_key", {"target_user_id": target_user_id}, True)
            return True, new_api_key
        else:
            return False, None

    def get_user_permissions(
        self, target_user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Obtém informações de permissões de um usuário.

        Args:
            target_user_id: ID do usuário (usa o atual se None)

        Returns:
            Dicionário com informações de permissões
        """
        user_id = target_user_id or self.user_id

        if user_id not in self.users:
            self.logger.warning(f"Usuário não encontrado: {user_id}")
            return {}

        user = self.users[user_id]
        permission_level = user.get("permission_level", "read")

        # Construir lista de operações permitidas
        allowed_operations = set()
        level_value = self.PERMISSION_LEVELS.get(permission_level, 0)

        for level, operations in self.OPERATIONS.items():
            level_req_value = self.PERMISSION_LEVELS.get(level, 0)
            if level_req_value <= level_value:
                allowed_operations.update(operations)

        return {
            "user_id": user_id,
            "permission_level": permission_level,
            "enabled": user.get("enabled", False),
            "allowed_operations": sorted(list(allowed_operations)),
        }

    def validate_operation(
        self, operation: str, params: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Valida se uma operação pode ser executada.

        Args:
            operation: Nome da operação
            params: Parâmetros da operação

        Returns:
            Tupla (permitido, mensagem)
        """
        # Verificar permissão básica
        if not self.has_permission(operation):
            return False, f"Sem permissão para executar a operação: {operation}"

        # Regras de validação específicas para operações sensíveis
        if operation == "create_pipeline":
            # Exemplo: validar nome de pipeline
            if "name" in params and len(params["name"]) < 3:
                return False, "Nome de pipeline muito curto"

        elif operation == "install_components":
            # Exemplo: validar origem de componentes
            if "source" in params and not params["source"].startswith("trusted_"):
                return False, "Fonte de componentes não confiável"

        # Registrar operação validada
        self._log_action(
            "validate_operation", {"operation": operation, "params": params}, True
        )

        return True, "Operação permitida"
