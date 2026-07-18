#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Formatadores de prompts para diferentes modelos de linguagem.

Este módulo fornece classes para formatar prompts de acordo com os
requisitos de diferentes arquiteturas de LLM como Qwen, Llama, etc.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Union


class BasePromptFormatter:
    """Classe base para formatadores de prompts."""

    def format_prompt(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Formata um prompt para o modelo.

        Args:
            user_input: Entrada do usuário
            system_prompt: Instrução do sistema (opcional)
            chat_history: Histórico de conversas (opcional)

        Returns:
            str: Prompt formatado
        """
        raise NotImplementedError("Método não implementado na classe base")


class QwenPromptFormatter(BasePromptFormatter):
    """
    Formatador de prompts para modelos Qwen.

    Compatível com modelos Qwen 1.5 e 2.0/2.5. O formato utiliza
    os tokens <|im_start|> e <|im_end|> para demarcar mensagens.
    """

    def format_prompt(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Formata um prompt para modelos Qwen.

        Args:
            user_input: Entrada do usuário
            system_prompt: Instrução do sistema (opcional)
            chat_history: Histórico de conversas (opcional)

        Returns:
            str: Prompt formatado para Qwen
        """
        messages = []

        # Adicionar instrução do sistema se fornecida
        if system_prompt:
            messages.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

        # Adicionar histórico de conversas
        if chat_history:
            for message in chat_history:
                role = message.get("role", "")
                content = message.get("content", "")

                if role and content:
                    # Formato: <|im_start|>role\ncontent<|im_end|>
                    messages.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # Adicionar a entrada do usuário atual
        messages.append(f"<|im_start|>user\n{user_input}<|im_end|>")

        # Preparar para a resposta do assistente
        messages.append("<|im_start|>assistant\n")

        # Juntar tudo em um único prompt
        return "\n".join(messages)


class Qwen2PromptFormatter(QwenPromptFormatter):
    """
    Formatador de prompts específico para Qwen 2.0 e 2.5.

    Implementa pequenas variações se necessário para versões mais recentes.
    """

    pass


class LlamaPromptFormatter(BasePromptFormatter):
    """
    Formatador de prompts para modelos Llama.

    Compatível com Llama 2 e similares, usando o formato de instruções
    [INST] e [/INST].
    """

    def format_prompt(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Formata um prompt para modelos Llama.

        Args:
            user_input: Entrada do usuário
            system_prompt: Instrução do sistema (opcional)
            chat_history: Histórico de conversas (opcional)

        Returns:
            str: Prompt formatado para Llama
        """
        messages = []

        # Cabeçalho do sistema
        header = ""
        if system_prompt:
            header = f"<s>[INST] {system_prompt} [/INST]\n\n"

        # Adicionar histórico de conversas
        conversation = ""
        if chat_history:
            for message in chat_history:
                role = message.get("role", "")
                content = message.get("content", "")

                if role == "user":
                    conversation += f"[INST] {content} [/INST]\n"
                elif role == "assistant":
                    conversation += f"{content}\n\n"

        # Adicionar a pergunta atual
        conversation += f"[INST] {user_input} [/INST]\n"

        # Formatar tudo junto
        full_prompt = header + conversation

        return full_prompt


def get_formatter_for_model(model_name: str) -> BasePromptFormatter:
    """
    Retorna o formatador apropriado para o modelo especificado.

    Args:
        model_name: Nome do modelo ou arquivo de modelo

    Returns:
        BasePromptFormatter: Formatador adequado para o modelo
    """
    model_name = model_name.lower()

    if "qwen2" in model_name or "qwen-2" in model_name or "qwen 2" in model_name:
        return Qwen2PromptFormatter()
    elif "qwen" in model_name:
        return QwenPromptFormatter()
    elif "llama" in model_name:
        return LlamaPromptFormatter()
    else:
        # Padrão para outros modelos
        return QwenPromptFormatter()  # Usando Qwen como padrão
