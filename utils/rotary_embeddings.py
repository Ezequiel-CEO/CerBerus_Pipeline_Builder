#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rotary Embeddings
================

Implementação otimizada de Rotary Position Embeddings (RoPE).
Suporta diferentes formatos de posicionamento e é otimizado para GPU.
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from enum import Enum


class RotaryFormat(Enum):
    """Formatos suportados para RoPE."""

    STANDARD = "standard"  # Formato padrão do paper original
    SCALED = "scaled"  # Com escala adaptativa
    DYNAMIC = "dynamic"  # Com base dinâmica
    HYBRID = "hybrid"  # Combinação de formatos


@dataclass
class RotaryConfig:
    """Configurações para Rotary Embeddings."""

    format: RotaryFormat = RotaryFormat.HYBRID
    dim: int = 64
    base: int = 10000
    scale: float = 1.0
    max_position: int = 2048
    dynamic_scale: bool = True
    interleaved: bool = True
    scaling_factor: float = 1.0


class RotaryEmbeddings(torch.nn.Module):
    """
    Implementação otimizada de Rotary Position Embeddings.
    Baseado no paper "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """

    def __init__(self, config: RotaryConfig):
        super().__init__()
        self.config = config

        # Validar configurações
        assert self.config.dim % 2 == 0, "Dimensão deve ser par"

        # Inicializar parâmetros
        self._init_rotation_matrices()

    def _init_rotation_matrices(self):
        """Inicializa matrizes de rotação."""

        # Calcular frequências de rotação
        if self.config.format == RotaryFormat.STANDARD:
            # Formato padrão do paper
            inv_freq = 1.0 / (
                self.config.base
                ** (torch.arange(0, self.config.dim, 2).float() / self.config.dim)
            )

        elif self.config.format == RotaryFormat.SCALED:
            # Formato com escala adaptativa
            inv_freq = self.config.scale * (
                1.0
                / (
                    self.config.base
                    ** (torch.arange(0, self.config.dim, 2).float() / self.config.dim)
                )
            )

        elif self.config.format == RotaryFormat.DYNAMIC:
            # Formato com base dinâmica
            base = self.config.base * (
                1.0 + torch.arange(0, self.config.dim, 2).float() / self.config.dim
            )
            inv_freq = 1.0 / base

        elif self.config.format == RotaryFormat.HYBRID:
            # Formato híbrido
            base = self.config.base * (
                1.0
                + self.config.scaling_factor
                * torch.arange(0, self.config.dim, 2).float()
                / self.config.dim
            )
            inv_freq = self.config.scale / base

        self.register_buffer("inv_freq", inv_freq)

    def _get_rotation_matrix(
        self, seq_len: int, device: torch.device, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gera matrizes de rotação.

        Args:
            seq_len: Comprimento da sequência
            device: Device para tensores
            offset: Offset de posição

        Returns:
            Tuple com cos e sin das rotações
        """
        # Gerar posições
        pos = torch.arange(seq_len, device=device) + offset

        # Calcular ângulos
        t = torch.outer(pos, self.inv_freq).float()

        # Aplicar escala dinâmica se configurado
        if self.config.dynamic_scale:
            scale = 1.0 / (1.0 + torch.log1p(pos.float() / self.config.max_position))
            t = t * scale.unsqueeze(-1)

        # Calcular rotações
        freqs = torch.cat([t, t], dim=-1)
        emb = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

        if self.config.interleaved:
            emb = emb.view(seq_len, -1, 2)
            cos_rot = emb[..., 0]
            sin_rot = emb[..., 1]
        else:
            split = emb.shape[-1] // 2
            cos_rot = emb[..., :split]
            sin_rot = emb[..., split:]

        return cos_rot, sin_rot

    def _apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Aplica rotações ao tensor de entrada.

        Args:
            x: Tensor de entrada [batch_size, seq_len, num_heads, head_dim]
            cos: Cossenos das rotações [seq_len, head_dim]
            sin: Senos das rotações [seq_len, head_dim]

        Returns:
            Tensor com rotações aplicadas
        """
        # Separar dimensões pares e ímpares
        x_split = x.view(*x.shape[:-1], -1, 2)
        x_even = x_split[..., 0]
        x_odd = x_split[..., 1]

        # Aplicar rotações
        x_rotated = torch.stack(
            [x_even * cos - x_odd * sin, x_odd * cos + x_even * sin], dim=-1
        )

        return x_rotated.flatten(-2)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Forward pass do módulo.

        Args:
            x: Tensor de entrada [batch_size, seq_len, num_heads, head_dim]
            offset: Offset de posição opcional

        Returns:
            Tensor com embeddings posicionais aplicados
        """
        seq_len = x.shape[1]
        cos_rot, sin_rot = self._get_rotation_matrix(seq_len, x.device, offset)

        # Expandir dimensões para broadcast
        cos_rot = cos_rot.view(seq_len, 1, -1)
        sin_rot = sin_rot.view(seq_len, 1, -1)

        # Aplicar rotações
        return self._apply_rotary(x, cos_rot, sin_rot)

    def extra_repr(self) -> str:
        """Representação string com parâmetros."""
        return (
            f"format={self.config.format.value}, "
            f"dim={self.config.dim}, "
            f"base={self.config.base}, "
            f"scale={self.config.scale}"
        )
