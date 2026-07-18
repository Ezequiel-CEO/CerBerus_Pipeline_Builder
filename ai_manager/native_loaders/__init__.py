# -*- coding: utf-8 -*-
"""
Loaders nativos para diferentes formatos de modelo
Implementações 100% nativas sem dependências externas
"""

from typing import List

# Lista de loaders disponíveis
AVAILABLE_LOADERS = ["gguf"]


def get_available_loaders() -> List[str]:
    """Retorna a lista de loaders disponíveis"""
    return AVAILABLE_LOADERS.copy()
