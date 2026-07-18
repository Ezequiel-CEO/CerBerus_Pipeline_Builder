#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flash Attention 2.0
==================

Implementação otimizada do Flash Attention 2.0 baseada no paper da OpenAI/Tri Dao.
Inclui otimizações de memória e computação para atenção eficiente em GPUs.
"""

import torch
import triton
import triton.language as tl
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FlashAttentionConfig:
    """Configurações para Flash Attention."""

    max_seq_length: int = 8192
    head_dim: int = 64
    num_heads: int = 32
    dropout_p: float = 0.0
    causal: bool = True
    block_size: int = 256
    num_warps: int = 8


@triton.jit
def _flash_attention_forward(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    sm_scale,
    stride_q_b,
    stride_q_h,
    stride_q_s,
    stride_k_b,
    stride_k_h,
    stride_k_s,
    stride_v_b,
    stride_v_h,
    stride_v_s,
    stride_o_b,
    stride_o_h,
    stride_o_s,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Kernel otimizado do Flash Attention forward pass."""

    # Índices para paralelização
    pid = tl.program_id(0)
    num_pipe = tl.cdiv(seq_len, BLOCK_SIZE)

    # Carregar query block
    q_block_ptr = q_ptr + pid * stride_q_b
    q_block = tl.load(q_block_ptr)

    # Inicializar acumuladores
    o_scale = tl.zeros([BLOCK_SIZE, head_dim], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop sobre blocos de key/value
    for block_idx in range(num_pipe):
        # Carregar key/value blocks
        k_block_ptr = k_ptr + block_idx * BLOCK_SIZE * stride_k_s
        v_block_ptr = v_ptr + block_idx * BLOCK_SIZE * stride_v_s

        k_block = tl.load(k_block_ptr)
        v_block = tl.load(v_block_ptr)

        # Computar scores de atenção
        scores = tl.dot(q_block, k_block.transpose())
        scores = scores * sm_scale

        # Aplicar máscara causal se necessário
        if CAUSAL:
            scores = tl.where(
                tl.arange(0, BLOCK_SIZE)[:, None] >= tl.arange(0, BLOCK_SIZE)[None, :],
                scores,
                float("-inf"),
            )

        # Atualizar acumuladores
        m_i_new = tl.maximum(m_i, tl.max(scores, 1))
        l_i_new = tl.exp(m_i - m_i_new[:, None]) * l_i
        p = tl.exp(scores - m_i_new[:, None])
        l_i_new += tl.sum(p, 1)

        # Atualizar output
        o_scale = tl.exp(m_i - m_i_new[:, None]) * o_scale + tl.dot(p, v_block)

        m_i = m_i_new
        l_i = l_i_new

    # Normalizar e salvar output
    o_ptr = o_ptr + pid * stride_o_b
    o_scale = o_scale / l_i[:, None]
    tl.store(o_ptr, o_scale)


class FlashAttention(torch.nn.Module):
    """Módulo PyTorch para Flash Attention 2.0."""

    def __init__(self, config: FlashAttentionConfig):
        super().__init__()
        self.config = config

        # Validar configurações
        assert self.config.head_dim <= 128, "head_dim deve ser <= 128 para eficiência"
        assert self.config.max_seq_length <= 8192, "seq_length deve ser <= 8192"

        # Calcular escala para scores de atenção
        self.scale = 1.0 / (self.config.head_dim**0.5)

        # Inicializar dropout
        self.dropout = torch.nn.Dropout(self.config.dropout_p)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass do Flash Attention.

        Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_heads, seq_len, head_dim]
            v: Value tensor [batch_size, num_heads, seq_len, head_dim]
            mask: Máscara opcional [batch_size, num_heads, seq_len, seq_len]

        Returns:
            Tuple com output tensor e scores de atenção
        """
        batch_size = q.shape[0]
        seq_len = q.shape[2]

        # Alocar memória para output
        output = torch.empty_like(q)

        # Configurar grid para kernel
        grid = (batch_size * self.config.num_heads,)

        # Lançar kernel forward
        _flash_attention_forward[grid](
            q,
            k,
            v,
            output,
            self.scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            batch_size,
            self.config.num_heads,
            seq_len,
            self.config.head_dim,
            BLOCK_SIZE=self.config.block_size,
            CAUSAL=self.config.causal,
            num_warps=self.config.num_warps,
        )

        # Aplicar dropout se necessário
        if self.training and self.config.dropout_p > 0:
            output = self.dropout(output)

        return output

    def extra_repr(self) -> str:
        """Representação string com parâmetros."""
        return (
            f"head_dim={self.config.head_dim}, "
            f"num_heads={self.config.num_heads}, "
            f"dropout={self.config.dropout_p}, "
            f"causal={self.config.causal}"
        )
