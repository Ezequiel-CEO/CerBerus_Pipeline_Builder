#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sparse Attention
===============

Implementação de Sparse Attention com múltiplos padrões de esparsidade:
- Strided
- Fixed
- Random
- Longformer
- BigBird
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
from enum import Enum


class SparsityPattern(Enum):
    """Padrões de esparsidade suportados."""

    STRIDED = "strided"
    FIXED = "fixed"
    RANDOM = "random"
    LONGFORMER = "longformer"
    BIGBIRD = "bigbird"


@dataclass
class SparseAttentionConfig:
    """Configurações para Sparse Attention."""

    pattern: SparsityPattern = SparsityPattern.BIGBIRD
    num_heads: int = 32
    head_dim: int = 64
    block_size: int = 64
    num_random_blocks: int = 3
    num_sliding_window_blocks: int = 3
    num_global_tokens: int = 2
    seed: int = 42
    dropout_p: float = 0.1


class SparseAttention(torch.nn.Module):
    """
    Implementação de Sparse Attention com múltiplos padrões.
    Baseado nos papers:
    - Longformer
    - BigBird
    - Sparse Transformer
    """

    def __init__(self, config: SparseAttentionConfig):
        super().__init__()
        self.config = config

        # Inicializar gerador de números aleatórios
        self.rng = np.random.RandomState(self.config.seed)

        # Inicializar dropout
        self.dropout = torch.nn.Dropout(self.config.dropout_p)

        # Calcular escala para scores
        self.scale = 1.0 / (self.config.head_dim**0.5)

    def _create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Cria máscara de esparsidade baseada no padrão selecionado.

        Args:
            seq_len: Comprimento da sequência
            device: Device para tensor

        Returns:
            Tensor booleano com máscara de esparsidade
        """
        num_blocks = seq_len // self.config.block_size
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

        if self.config.pattern == SparsityPattern.STRIDED:
            # Padrão strided: atende a cada k-ésimo token
            stride = self.config.block_size
            for i in range(seq_len):
                start = (i // stride) * stride
                end = min(start + stride, seq_len)
                mask[i, start:end] = True

        elif self.config.pattern == SparsityPattern.FIXED:
            # Padrão fixo: blocos diagonais + alguns blocos fixos
            for i in range(num_blocks):
                # Bloco diagonal
                start_row = i * self.config.block_size
                end_row = start_row + self.config.block_size
                start_col = start_row
                end_col = end_row
                mask[start_row:end_row, start_col:end_col] = True

                # Blocos fixos adicionais
                if i + 1 < num_blocks:
                    mask[
                        start_row:end_row, end_col : end_col + self.config.block_size
                    ] = True

        elif self.config.pattern == SparsityPattern.RANDOM:
            # Padrão aleatório: blocos diagonais + blocos aleatórios
            for i in range(num_blocks):
                # Bloco diagonal
                start_row = i * self.config.block_size
                end_row = start_row + self.config.block_size
                start_col = start_row
                end_col = end_row
                mask[start_row:end_row, start_col:end_col] = True

                # Blocos aleatórios
                random_blocks = self.rng.choice(
                    num_blocks, size=self.config.num_random_blocks, replace=False
                )
                for j in random_blocks:
                    start_col = j * self.config.block_size
                    end_col = start_col + self.config.block_size
                    mask[start_row:end_row, start_col:end_col] = True

        elif self.config.pattern == SparsityPattern.LONGFORMER:
            # Padrão Longformer: sliding window + global tokens
            window_size = self.config.num_sliding_window_blocks * self.config.block_size

            # Sliding window
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2)
                mask[i, start:end] = True

            # Global tokens
            mask[: self.config.num_global_tokens, :] = True
            mask[:, : self.config.num_global_tokens] = True

        elif self.config.pattern == SparsityPattern.BIGBIRD:
            # Padrão BigBird: random + sliding window + global

            # Global tokens
            mask[: self.config.num_global_tokens, :] = True
            mask[:, : self.config.num_global_tokens] = True

            for i in range(num_blocks):
                start_row = i * self.config.block_size
                end_row = start_row + self.config.block_size

                # Sliding window
                for w in range(
                    -self.config.num_sliding_window_blocks,
                    self.config.num_sliding_window_blocks + 1,
                ):
                    if 0 <= i + w < num_blocks:
                        start_col = (i + w) * self.config.block_size
                        end_col = start_col + self.config.block_size
                        mask[start_row:end_row, start_col:end_col] = True

                # Random blocks
                random_blocks = self.rng.choice(
                    num_blocks, size=self.config.num_random_blocks, replace=False
                )
                for j in random_blocks:
                    start_col = j * self.config.block_size
                    end_col = start_col + self.config.block_size
                    mask[start_row:end_row, start_col:end_col] = True

        return mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass do Sparse Attention.

        Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_heads, seq_len, head_dim]
            v: Value tensor [batch_size, num_heads, seq_len, head_dim]
            mask: Máscara adicional opcional

        Returns:
            Tuple com output tensor e scores de atenção
        """
        batch_size = q.shape[0]
        num_heads = q.shape[1]
        seq_len = q.shape[2]

        # Criar máscara de esparsidade
        sparsity_mask = self._create_sparse_mask(seq_len, q.device)
        sparsity_mask = sparsity_mask.unsqueeze(0).unsqueeze(0)
        sparsity_mask = sparsity_mask.expand(batch_size, num_heads, -1, -1)

        # Combinar com máscara adicional se fornecida
        if mask is not None:
            sparsity_mask = sparsity_mask & mask

        # Computar scores de atenção
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Aplicar máscara de esparsidade
        scores = scores.masked_fill(~sparsity_mask, float("-inf"))

        # Softmax e dropout
        attention_probs = F.softmax(scores, dim=-1)
        if self.training:
            attention_probs = self.dropout(attention_probs)

        # Computar output
        output = torch.matmul(attention_probs, v)

        return output, attention_probs

    def extra_repr(self) -> str:
        """Representação string com parâmetros."""
        return (
            f"pattern={self.config.pattern.value}, "
            f"num_heads={self.config.num_heads}, "
            f"head_dim={self.config.head_dim}, "
            f"block_size={self.config.block_size}"
        )
