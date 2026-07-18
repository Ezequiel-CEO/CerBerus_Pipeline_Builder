"""
Módulo de armazenamento para o Pipeline Builder
"""

from typing import Any, Dict, List, Optional


class StorageManager:
    def __init__(self):
        self.data = {}

    def save(self, key: str, value: Any) -> bool:
        """Salva um valor no armazenamento"""
        try:
            self.data[key] = value
            return True
        except Exception:
            return False

    def load(self, key: str, default: Any = None) -> Any:
        """Carrega um valor do armazenamento"""
        return self.data.get(key, default)

    def delete(self, key: str) -> bool:
        """Remove um valor do armazenamento"""
        try:
            del self.data[key]
            return True
        except KeyError:
            return False


# Instância global do gerenciador de armazenamento
storage_manager = StorageManager()

__all__ = ["StorageManager", "storage_manager"]
