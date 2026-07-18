#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CUDA Kernels: Implementações de operações de tensor otimizadas para GPU.

Fornece implementações otimizadas de operações comuns em modelos de IA,
com fallback para CPU usando NumPy.
"""

import os
import gc
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

# Configuração de logging
from cerberus_api.utils.logging_config import get_logger

logger = get_logger("cuda_kernels")


class CUDAKernels:
    """
    Implementações de operações otimizadas para GPU com fallback para CPU.

    Esta classe fornece operações de tensor optimizadas para GPU usando
    o framework interno do CerBerus, com fallback automático para CPU usando
    NumPy quando necessário.
    """

    def __init__(self, gpu_manager=None):
        """
        Inicializa os kernels CUDA.

        Args:
            gpu_manager: Instância do GPUManager para gerenciar memória
        """
        self.logger = logger
        self.gpu_manager = gpu_manager
        self.use_gpu = gpu_manager is not None and gpu_manager.use_gpu

        # Flag para rastrear erros
        self.had_errors = False

        # Exibir informações sobre o ambiente
        if self.use_gpu:
            self.logger.info("CUDA Kernels inicializados com suporte a GPU")
        else:
            self.logger.info("CUDA Kernels inicializados em modo CPU (NumPy)")

    # ==========================================================================
    # Operações básicas de tensor
    # ==========================================================================

    def matmul(self, a, b):
        """
        Multiplicação de matrizes otimizada.

        Args:
            a: Primeira matriz
            b: Segunda matriz

        Returns:
            Resultado da multiplicação
        """
        if self.use_gpu and not self.had_errors:
            try:
                return self.gpu_manager.matmul(a, b)
            except Exception as e:
                self.logger.warning(f"Erro em matmul na GPU: {e}. Caindo para CPU.")
                self.had_errors = True

        # Fallback para NumPy
        a_cpu = a
        b_cpu = b

        # Converter para CPU se necessário
        if self.use_gpu:
            a_cpu = (
                a
                if not self.gpu_manager.is_on_device(a)
                else self.gpu_manager.to_host(a)
            )
            b_cpu = (
                b
                if not self.gpu_manager.is_on_device(b)
                else self.gpu_manager.to_host(b)
            )

        return np.matmul(a_cpu, b_cpu)

    def add(self, a, b):
        """
        Adição de tensores otimizada.

        Args:
            a: Primeiro tensor
            b: Segundo tensor

        Returns:
            Resultado da adição
        """
        if self.use_gpu and not self.had_errors:
            try:
                # Verificar se os tensores já estão na GPU
                a_gpu = (
                    a
                    if self.gpu_manager.is_on_device(a)
                    else self.gpu_manager.to_device(a)
                )
                b_gpu = (
                    b
                    if self.gpu_manager.is_on_device(b)
                    else self.gpu_manager.to_device(b)
                )

                # Adição na GPU
                if hasattr(self.gpu_manager.cerberus_cupy, "add"):
                    return self.gpu_manager.cerberus_cupy.add(a_gpu, b_gpu)
                else:
                    # Implementação básica se não tiver método específico
                    return a_gpu + b_gpu

            except Exception as e:
                self.logger.warning(f"Erro em add na GPU: {e}. Caindo para CPU.")
                self.had_errors = True

        # Fallback para NumPy
        a_cpu = a
        b_cpu = b

        # Converter para CPU se necessário
        if self.use_gpu:
            a_cpu = (
                a
                if not self.gpu_manager.is_on_device(a)
                else self.gpu_manager.to_host(a)
            )
            b_cpu = (
                b
                if not self.gpu_manager.is_on_device(b)
                else self.gpu_manager.to_host(b)
            )

        return np.add(a_cpu, b_cpu)

    def mul(self, a, b):
        """
        Multiplicação elemento a elemento otimizada.

        Args:
            a: Primeiro tensor
            b: Segundo tensor

        Returns:
            Resultado da multiplicação elemento a elemento
        """
        if self.use_gpu and not self.had_errors:
            try:
                # Verificar se os tensores já estão na GPU
                a_gpu = (
                    a
                    if self.gpu_manager.is_on_device(a)
                    else self.gpu_manager.to_device(a)
                )
                b_gpu = (
                    b
                    if self.gpu_manager.is_on_device(b)
                    else self.gpu_manager.to_device(b)
                )

                # Multiplicação na GPU
                if hasattr(self.gpu_manager.cerberus_cupy, "multiply"):
                    return self.gpu_manager.cerberus_cupy.multiply(a_gpu, b_gpu)
                else:
                    # Implementação básica se não tiver método específico
                    return a_gpu * b_gpu

            except Exception as e:
                self.logger.warning(f"Erro em mul na GPU: {e}. Caindo para CPU.")
                self.had_errors = True

        # Fallback para NumPy
        a_cpu = a
        b_cpu = b

        # Converter para CPU se necessário
        if self.use_gpu:
            a_cpu = (
                a
                if not self.gpu_manager.is_on_device(a)
                else self.gpu_manager.to_host(a)
            )
            b_cpu = (
                b
                if not self.gpu_manager.is_on_device(b)
                else self.gpu_manager.to_host(b)
            )

        return np.multiply(a_cpu, b_cpu)

    # ==========================================================================
    # Funções de ativação e operações não-lineares
    # ==========================================================================

    def relu(self, x):
        """
        ReLU (Rectified Linear Unit) otimizada.

        Args:
            x: Tensor de entrada

        Returns:
            Tensor após aplicação de ReLU
        """
        if self.use_gpu and not self.had_errors:
            try:
                # Verificar se o tensor já está na GPU
                x_gpu = (
                    x
                    if self.gpu_manager.is_on_device(x)
                    else self.gpu_manager.to_device(x)
                )

                # ReLU na GPU
                if hasattr(self.gpu_manager.cerberus_cupy, "maximum"):
                    return self.gpu_manager.cerberus_cupy.maximum(0, x_gpu)
                else:
                    # Implementação básica
                    zeros = self.gpu_manager.to_device(np.zeros_like(np.array(x)))
                    return self.gpu_manager.cerberus_cupy.maximum(zeros, x_gpu)

            except Exception as e:
                self.logger.warning(f"Erro em relu na GPU: {e}. Caindo para CPU.")
                self.had_errors = True

        # Fallback para NumPy
        x_cpu = x

        # Converter para CPU se necessário
        if self.use_gpu:
            x_cpu = (
                x
                if not self.gpu_manager.is_on_device(x)
                else self.gpu_manager.to_host(x)
            )

        return np.maximum(0, x_cpu)

    def gelu(self, x):
        """
        GELU (Gaussian Error Linear Unit) otimizada.

        Args:
            x: Tensor de entrada

        Returns:
            Tensor após aplicação de GELU
        """
        if self.use_gpu and not self.had_errors:
            try:
                # Verificar se o tensor já está na GPU
                x_gpu = (
                    x
                    if self.gpu_manager.is_on_device(x)
                    else self.gpu_manager.to_device(x)
                )

                # GELU na GPU - implementação aproximada
                cupy = self.gpu_manager.cerberus_cupy

                # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                coef = np.sqrt(2 / np.pi)
                if hasattr(cupy, "power"):
                    x_cubed = cupy.power(x_gpu, 3)
                else:
                    x_cubed = x_gpu * x_gpu * x_gpu

                inner = coef * (x_gpu + 0.044715 * x_cubed)

                if hasattr(cupy, "tanh"):
                    tanh_term = cupy.tanh(inner)
                else:
                    # Aproximação menos precisa
                    exp_pos = cupy.exp(inner)
                    exp_neg = cupy.exp(-inner)
                    tanh_term = (exp_pos - exp_neg) / (exp_pos + exp_neg)

                return 0.5 * x_gpu * (1 + tanh_term)

            except Exception as e:
                self.logger.warning(f"Erro em gelu na GPU: {e}. Caindo para CPU.")
                self.had_errors = True

        # Fallback para NumPy
        x_cpu = x

        # Converter para CPU se necessário
        if self.use_gpu:
            x_cpu = (
                x
                if not self.gpu_manager.is_on_device(x)
                else self.gpu_manager.to_host(x)
            )

        # Implementação de GELU em NumPy
        coef = np.sqrt(2 / np.pi)
        return (
            0.5 * x_cpu * (1 + np.tanh(coef * (x_cpu + 0.044715 * np.power(x_cpu, 3))))
        )

    def softmax(self, x, axis=-1):
        """
        Softmax otimizada.

        Args:
            x: Tensor de entrada
            axis: Eixo para aplicar softmax

        Returns:
            Tensor após aplicação de softmax
        """
        if self.use_gpu:
            return self.gpu_manager.softmax(x, axis=axis)

        # Caso CPU
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    # ==========================================================================
    # Operações para redes neurais
    # ==========================================================================

    def layer_norm(self, x, weight, bias, eps=1e-5):
        """
        Normalização de camada (Layer Normalization) otimizada.

        Args:
            x: Tensor de entrada
            weight: Parâmetro gamma
            bias: Parâmetro beta
            eps: Epsilon para estabilidade numérica

        Returns:
            Tensor normalizado
        """
        if self.use_gpu and not self.had_errors:
            try:
                # Verificar se os tensores já estão na GPU
                x_gpu = (
                    x
                    if self.gpu_manager.is_on_device(x)
                    else self.gpu_manager.to_device(x)
                )
                weight_gpu = (
                    weight
                    if self.gpu_manager.is_on_device(weight)
                    else self.gpu_manager.to_device(weight)
                )
                bias_gpu = (
                    bias
                    if self.gpu_manager.is_on_device(bias)
                    else self.gpu_manager.to_device(bias)
                )

                cupy = self.gpu_manager.cerberus_cupy

                # Calcular média e variância
                mean = cupy.mean(x_gpu, axis=-1, keepdims=True)
                var = cupy.var(x_gpu, axis=-1, keepdims=True)

                # Normalizar
                x_norm = (x_gpu - mean) / cupy.sqrt(var + eps)

                # Aplicar parâmetros
                return x_norm * weight_gpu + bias_gpu

            except Exception as e:
                self.logger.warning(f"Erro em layer_norm na GPU: {e}. Caindo para CPU.")
                self.had_errors = True

        # Fallback para NumPy
        x_cpu = x
        weight_cpu = weight
        bias_cpu = bias

        # Converter para CPU se necessário
        if self.use_gpu:
            x_cpu = (
                x
                if not self.gpu_manager.is_on_device(x)
                else self.gpu_manager.to_host(x)
            )
            weight_cpu = (
                weight
                if not self.gpu_manager.is_on_device(weight)
                else self.gpu_manager.to_host(weight)
            )
            bias_cpu = (
                bias
                if not self.gpu_manager.is_on_device(bias)
                else self.gpu_manager.to_host(bias)
            )

        # Layer norm em NumPy
        mean = np.mean(x_cpu, axis=-1, keepdims=True)
        var = np.var(x_cpu, axis=-1, keepdims=True)
        x_norm = (x_cpu - mean) / np.sqrt(var + eps)
        return x_norm * weight_cpu + bias_cpu

    def attention(self, query, key, value, mask=None, scale=None):
        """
        Mecanismo de atenção otimizado.

        Args:
            query: Tensor de query
            key: Tensor de key
            value: Tensor de value
            mask: Máscara opcional (para atenção causal/padding)
            scale: Fator de escala opcional

        Returns:
            Tensor de saída da atenção
        """
        if scale is None:
            scale = 1.0 / np.sqrt(query.shape[-1])

        if self.use_gpu and not self.had_errors:
            try:
                # Verificar se os tensores já estão na GPU
                q_gpu = (
                    query
                    if self.gpu_manager.is_on_device(query)
                    else self.gpu_manager.to_device(query)
                )
                k_gpu = (
                    key
                    if self.gpu_manager.is_on_device(key)
                    else self.gpu_manager.to_device(key)
                )
                v_gpu = (
                    value
                    if self.gpu_manager.is_on_device(value)
                    else self.gpu_manager.to_device(value)
                )

                # Computar scores de atenção
                # (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len)
                # -> (batch, heads, seq_len, seq_len)
                k_t = self.gpu_manager.cerberus_cupy.transpose(k_gpu, (0, 1, 3, 2))
                scores = self.gpu_manager.matmul(q_gpu, k_t) * scale

                # Aplicar máscara se fornecida
                if mask is not None:
                    mask_gpu = (
                        mask
                        if self.gpu_manager.is_on_device(mask)
                        else self.gpu_manager.to_device(mask)
                    )
                    scores = scores + mask_gpu

                # Aplicar softmax
                attn_weights = self.softmax(scores, axis=-1)

                # Aplicar atenção aos valores
                # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, head_dim)
                # -> (batch, heads, seq_len, head_dim)
                output = self.gpu_manager.matmul(attn_weights, v_gpu)

                return output

            except Exception as e:
                self.logger.warning(f"Erro em attention na GPU: {e}. Caindo para CPU.")
                self.had_errors = True

        # Fallback para NumPy
        q = query
        k = key
        v = value

        # Converter para CPU se necessário
        if self.use_gpu:
            q = (
                query
                if not self.gpu_manager.is_on_device(query)
                else self.gpu_manager.to_host(query)
            )
            k = (
                key
                if not self.gpu_manager.is_on_device(key)
                else self.gpu_manager.to_host(key)
            )
            v = (
                value
                if not self.gpu_manager.is_on_device(value)
                else self.gpu_manager.to_host(value)
            )
            if mask is not None:
                mask = (
                    mask
                    if not self.gpu_manager.is_on_device(mask)
                    else self.gpu_manager.to_host(mask)
                )

        # Computar scores de atenção
        k_t = np.transpose(k, (0, 1, 3, 2))
        scores = np.matmul(q, k_t) * scale

        # Aplicar máscara se fornecida
        if mask is not None:
            scores = scores + mask

        # Aplicar softmax
        attn_weights = self.softmax(scores, axis=-1)

        # Aplicar atenção aos valores
        output = np.matmul(attn_weights, v)

        return output

    # ==========================================================================
    # Utilidades e operações auxiliares
    # ==========================================================================

    def to_device(self, x):
        """
        Transfere um tensor para o dispositivo atual.

        Args:
            x: Tensor a transferir

        Returns:
            Tensor no dispositivo alvo
        """
        if self.use_gpu:
            return self.gpu_manager.to_device(x)
        return x

    def to_host(self, x):
        """
        Transfere um tensor para a CPU.

        Args:
            x: Tensor a transferir

        Returns:
            Tensor na CPU
        """
        if self.use_gpu:
            return self.gpu_manager.to_host(x)
        return x

    def clear_cache(self):
        """Limpa caches de memória."""
        if self.use_gpu:
            self.gpu_manager.clear_cache()
        else:
            gc.collect()

    def get_memory_info(self):
        """
        Obtém informações sobre o uso de memória.

        Returns:
            Dicionário com informações de memória
        """
        if self.use_gpu:
            return self.gpu_manager.get_memory_info()

        # Versão simplificada para CPU
        try:
            import psutil

            process = psutil.Process(os.getpid())
            ram_used = process.memory_info().rss / (1024 * 1024)  # em MB
            ram_total = psutil.virtual_memory().total / (1024 * 1024)  # em MB

            return {
                "ram_used_mb": ram_used,
                "ram_total_mb": ram_total,
                "ram_percent": (ram_used / ram_total) * 100,
                "vram_used_mb": 0,
                "vram_total_mb": 0,
                "vram_percent": 0,
            }
        except ImportError:
            return {
                "ram_used_mb": 0,
                "ram_total_mb": 0,
                "ram_percent": 0,
                "vram_used_mb": 0,
                "vram_total_mb": 0,
                "vram_percent": 0,
            }
