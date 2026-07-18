#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configurações e Kernels Triton
=============================

Kernels CUDA otimizados usando Triton para operações de alto desempenho.
"""

import triton
import triton.language as tl
import torch
from typing import Optional, Tuple


@triton.jit
def attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    stride_q_b,
    stride_q_h,
    stride_q_s,
    stride_k_b,
    stride_k_h,
    stride_k_s,
    stride_v_b,
    stride_v_h,
    stride_v_s,
    stride_out_b,
    stride_out_h,
    stride_out_s,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel otimizado para self-attention."""
    # Índices para paralelização
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_SIZE)
    num_pid_n = tl.cdiv(seq_len, BLOCK_SIZE)
    num_pid_in_group = pid // num_pid_m
    group_id = pid % num_pid_m

    # Offsets para acessos à memória
    offs_m = group_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = num_pid_in_group * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Carregar Q, K, V
    q = tl.load(q_ptr + offs_m[:, None] * stride_q_s)
    k = tl.load(k_ptr + offs_n[None, :] * stride_k_s)
    v = tl.load(v_ptr + offs_n[:, None] * stride_v_s)

    # Computar scores de atenção
    scores = tl.dot(q, k) / tl.sqrt(head_dim)
    scores = tl.softmax(scores)

    # Computar output
    out = tl.dot(scores, v)

    # Salvar resultado
    out_ptr = out_ptr + offs_m[:, None] * stride_out_s
    tl.store(out_ptr, out)


@triton.jit
def layer_norm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    stride_x_b,
    stride_x_f,
    stride_gamma,
    stride_beta,
    batch_size,
    hidden_size,
    eps: tl.constexpr,
):
    """Kernel otimizado para layer normalization."""
    # Índices
    pid = tl.program_id(0)

    # Carregar dados
    x = tl.load(x_ptr + pid * stride_x_b)
    gamma = tl.load(gamma_ptr)
    beta = tl.load(beta_ptr)

    # Computar média
    mean = tl.sum(x, axis=1) / hidden_size

    # Computar variância
    x_centered = x - mean[:, None]
    var = tl.sum(x_centered * x_centered, axis=1) / hidden_size

    # Normalizar
    x_norm = x_centered / tl.sqrt(var[:, None] + eps)

    # Aplicar gamma e beta
    out = gamma * x_norm + beta

    # Salvar resultado
    tl.store(out_ptr + pid * stride_x_b, out)


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Kernel otimizado para GELU activation."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Carregar dados
    x = tl.load(x_ptr + offsets, mask=mask)

    # Constantes para aproximação de GELU
    sqrt_2_pi = 2.5066282746310002
    coef = 0.044715

    # Computar GELU
    cdf = 0.5 * (1.0 + tl.tanh(sqrt_2_pi * (x + coef * x * x * x)))
    out = x * cdf

    # Salvar resultado
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def rotary_embedding_kernel(
    x_ptr, out_ptr, seq_len, hidden_size, base: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """Kernel otimizado para Rotary Position Embeddings."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Índices e máscaras
    pos = block_start + tl.arange(0, BLOCK_SIZE)
    dim = tl.arange(0, hidden_size)
    mask = pos < seq_len

    # Computar theta
    theta = base ** (-2.0 * (dim // 2) / hidden_size)

    # Carregar dados
    x = tl.load(x_ptr + pos[:, None] * hidden_size + dim[None, :], mask=mask[:, None])

    # Aplicar rotação
    x_rot = tl.zeros_like(x)
    x_rot[:, ::2] = x[:, ::2] * tl.cos(theta[None, ::2]) - x[:, 1::2] * tl.sin(
        theta[None, ::2]
    )
    x_rot[:, 1::2] = x[:, ::2] * tl.sin(theta[None, ::2]) + x[:, 1::2] * tl.cos(
        theta[None, ::2]
    )

    # Salvar resultado
    tl.store(
        out_ptr + pos[:, None] * hidden_size + dim[None, :], x_rot, mask=mask[:, None]
    )


class TritonKernelConfig:
    """Configurações para kernels Triton."""

    @staticmethod
    def get_optimal_block_size(n: int) -> int:
        """
        Calcula tamanho ótimo de bloco para kernels.

        Args:
            n: Número de elementos

        Returns:
            Tamanho do bloco otimizado
        """
        return min(max(32, (n + 255) // 256 * 256), 1024)

    @staticmethod
    def get_grid_blocks(n: int, block_size: int) -> int:
        """
        Calcula número de blocos para grid.

        Args:
            n: Número de elementos
            block_size: Tamanho do bloco

        Returns:
            Número de blocos
        """
        return (n + block_size - 1) // block_size

    @staticmethod
    def get_attention_configs(
        batch_size: int, num_heads: int, seq_len: int, head_dim: int
    ) -> Tuple[int, int]:
        """
        Configurações para kernel de attention.

        Args:
            batch_size: Tamanho do batch
            num_heads: Número de heads
            seq_len: Comprimento da sequência
            head_dim: Dimensão do head

        Returns:
            Tuple com tamanho do bloco e número de threads
        """
        block_size = min(seq_len, 256)
        num_warps = min(max(1, block_size // 32), 8)
        return block_size, num_warps

    @staticmethod
    def get_layer_norm_configs(batch_size: int, hidden_size: int) -> Tuple[int, int]:
        """
        Configurações para kernel de layer norm.

        Args:
            batch_size: Tamanho do batch
            hidden_size: Dimensão do modelo

        Returns:
            Tuple com tamanho do bloco e número de threads
        """
        block_size = min(hidden_size, 1024)
        num_warps = min(max(1, block_size // 32), 8)
        return block_size, num_warps


def launch_attention_kernel(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """
    Lança kernel de attention otimizado.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        num_heads: Número de attention heads
        head_dim: Dimensão de cada head

    Returns:
        Tensor de output da attention
    """
    batch_size = q.shape[0]
    seq_len = q.shape[1]

    # Alocar output
    output = torch.empty_like(q)

    # Configurar kernel
    block_size, num_warps = TritonKernelConfig.get_attention_configs(
        batch_size, num_heads, seq_len, head_dim
    )

    # Lançar kernel
    grid = (batch_size * num_heads,)
    attention_kernel[grid](
        q,
        k,
        v,
        output,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
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
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return output


def launch_layer_norm_kernel(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    Lança kernel de layer normalization otimizado.

    Args:
        x: Input tensor
        gamma: Peso de escala
        beta: Bias
        eps: Epsilon para estabilidade numérica

    Returns:
        Tensor normalizado
    """
    batch_size = x.shape[0]
    hidden_size = x.shape[-1]

    # Alocar output
    output = torch.empty_like(x)

    # Configurar kernel
    block_size, num_warps = TritonKernelConfig.get_layer_norm_configs(
        batch_size, hidden_size
    )

    # Lançar kernel
    grid = (batch_size,)
    layer_norm_kernel[grid](
        x,
        gamma,
        beta,
        output,
        x.stride(0),
        x.stride(1),
        gamma.stride(0),
        beta.stride(0),
        batch_size,
        hidden_size,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return output
