#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configurações e Otimizações JAX
==============================

Otimizações avançadas usando JAX para operações de alto desempenho.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
from jax.experimental import maps
from jax.experimental.pjit import pjit
from jax.experimental.maps import xmap
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Configurar JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")


class JAXOptimizer:
    """Otimizador avançado usando JAX."""

    def __init__(self, use_tpu: bool = False):
        """
        Inicializa otimizador JAX.

        Args:
            use_tpu: Se True, otimiza para TPU
        """
        self.use_tpu = use_tpu
        self.device_count = jax.device_count()
        self.devices = jax.devices()

        # Cache de funções compiladas
        self._compiled_fns = {}

    @staticmethod
    @jit
    def gelu_forward(x: jnp.ndarray) -> jnp.ndarray:
        """
        GELU forward pass otimizado.

        Args:
            x: Input array

        Returns:
            Output array
        """
        sqrt_2_pi = jnp.sqrt(2.0 / jnp.pi)
        cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_pi * (x + 0.044715 * jnp.power(x, 3))))
        return x * cdf

    @staticmethod
    @jit
    def gelu_backward(x: jnp.ndarray, grad_output: jnp.ndarray) -> jnp.ndarray:
        """
        GELU backward pass otimizado.

        Args:
            x: Input array
            grad_output: Gradiente do output

        Returns:
            Gradiente do input
        """
        return grad(lambda x: jnp.sum(JAXOptimizer.gelu_forward(x)))(x) * grad_output

    @staticmethod
    @jit
    def layer_norm(
        x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, eps: float = 1e-5
    ) -> jnp.ndarray:
        """
        Layer normalization otimizada.

        Args:
            x: Input array
            gamma: Peso de escala
            beta: Bias
            eps: Epsilon para estabilidade

        Returns:
            Array normalizado
        """
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / jnp.sqrt(var + eps) + beta

    @staticmethod
    @jit
    def softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        """
        Softmax otimizado.

        Args:
            x: Input array
            axis: Eixo para aplicar softmax

        Returns:
            Array com softmax aplicado
        """
        max_x = jnp.max(x, axis=axis, keepdims=True)
        exp_x = jnp.exp(x - max_x)
        return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    @jit
    def attention(
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        dropout_p: float = 0.0,
        is_training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Self-attention otimizada.

        Args:
            q: Query array
            k: Key array
            v: Value array
            mask: Máscara de atenção opcional
            dropout_p: Taxa de dropout
            is_training: Se está em modo de treino

        Returns:
            Tuple com output e scores de atenção
        """
        # Computar scores
        scores = jnp.matmul(q, k.transpose(-2, -1)) / jnp.sqrt(q.shape[-1])

        # Aplicar máscara se fornecida
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)

        # Softmax e dropout
        attention_probs = JAXOptimizer.softmax(scores)
        if is_training and dropout_p > 0:
            key = jax.random.PRNGKey(0)
            attention_probs = (
                jax.random.bernoulli(key, p=1 - dropout_p, shape=attention_probs.shape)
                * attention_probs
                / (1 - dropout_p)
            )

        # Computar output
        output = jnp.matmul(attention_probs, v)
        return output, attention_probs

    @staticmethod
    @jit
    def rotary_embedding(
        x: jnp.ndarray, seq_len: int, dim: int, base: float = 10000.0
    ) -> jnp.ndarray:
        """
        Rotary position embeddings otimizados.

        Args:
            x: Input array
            seq_len: Comprimento da sequência
            dim: Dimensão do modelo
            base: Base para cálculo das frequências

        Returns:
            Array com embeddings aplicados
        """
        # Gerar frequências
        freqs = jnp.arange(0, dim, 2)
        freqs = base ** (-freqs / dim)

        # Gerar posições
        pos = jnp.arange(seq_len)
        pos = pos[:, None] * freqs[None, :]

        # Calcular senos e cossenos
        sin = jnp.sin(pos)
        cos = jnp.cos(pos)

        # Aplicar rotação
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        # Retornar com rotação aplicada
        return jnp.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).reshape(
            x.shape
        )

    def parallelize(
        self,
        fn: callable,
        in_axes: Union[int, Tuple[int, ...]] = 0,
        out_axes: Union[int, Tuple[int, ...]] = 0,
    ) -> callable:
        """
        Paraleliza uma função usando vmap/pmap.

        Args:
            fn: Função a paralelizar
            in_axes: Eixos de entrada para paralelização
            out_axes: Eixos de saída para paralelização

        Returns:
            Função paralelizada
        """
        if self.device_count > 1:
            # Usar pmap para multi-GPU/TPU
            return pmap(fn, in_axes=in_axes, out_axes=out_axes)
        else:
            # Usar vmap para single-GPU
            return vmap(fn, in_axes=in_axes, out_axes=out_axes)

    def optimize_matmul(
        self, batch_size: int, seq_len: int, hidden_size: int
    ) -> callable:
        """
        Otimiza multiplicação de matrizes para dimensões específicas.

        Args:
            batch_size: Tamanho do batch
            seq_len: Comprimento da sequência
            hidden_size: Dimensão do modelo

        Returns:
            Função otimizada de matmul
        """
        # Chave para cache
        key = f"matmul_{batch_size}_{seq_len}_{hidden_size}"

        if key not in self._compiled_fns:
            # Definir função otimizada
            @jit
            def optimized_matmul(a, b):
                return jnp.matmul(a, b)

            # Compilar com shapes específicos
            dummy_a = jnp.ones((batch_size, seq_len, hidden_size))
            dummy_b = jnp.ones((batch_size, hidden_size, seq_len))

            # Pré-compilar e cachear
            self._compiled_fns[key] = optimized_matmul
            _ = optimized_matmul(dummy_a, dummy_b)

        return self._compiled_fns[key]

    def create_mesh(
        self, mesh_shape: Tuple[int, ...], axis_names: Tuple[str, ...]
    ) -> maps.Mesh:
        """
        Cria mesh para paralelismo de modelo/dados.

        Args:
            mesh_shape: Shape do mesh
            axis_names: Nomes dos eixos

        Returns:
            Mesh JAX configurado
        """
        devices = np.array(jax.devices()).reshape(mesh_shape)
        return maps.Mesh(devices, axis_names)

    def shard_params(
        self, params: Dict[str, jnp.ndarray], mesh: maps.Mesh, rules: Dict[str, str]
    ) -> Dict[str, jnp.ndarray]:
        """
        Sharda parâmetros do modelo entre devices.

        Args:
            params: Dicionário de parâmetros
            mesh: Mesh JAX
            rules: Regras de sharding

        Returns:
            Parâmetros shardados
        """
        with mesh:
            sharded_params = {
                k: pjit(
                    lambda x: x,
                    in_axis_resources=rules.get(k, None),
                    out_axis_resources=rules.get(k, None),
                )(v)
                for k, v in params.items()
            }
        return sharded_params

    def get_optimal_mesh(
        self, model_size: int, batch_size: int, num_layers: int
    ) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
        """
        Determina configuração ótima de mesh.

        Args:
            model_size: Tamanho do modelo
            batch_size: Tamanho do batch
            num_layers: Número de camadas

        Returns:
            Tuple com shape e nomes dos eixos
        """
        if self.device_count >= 8:
            # Para 8+ GPUs, usar paralelismo de modelo e dados
            return (2, self.device_count // 2), ("model", "batch")
        elif self.device_count >= 2:
            # Para 2-7 GPUs, usar apenas paralelismo de dados
            return (self.device_count,), ("batch",)
        else:
            # Para 1 GPU, não usar mesh
            return (1,), ("batch",)

    def get_optimal_rules(
        self, param_shapes: Dict[str, Tuple[int, ...]], mesh_axes: Tuple[str, ...]
    ) -> Dict[str, str]:
        """
        Determina regras ótimas de sharding.

        Args:
            param_shapes: Shapes dos parâmetros
            mesh_axes: Nomes dos eixos do mesh

        Returns:
            Dicionário com regras de sharding
        """
        rules = {}
        for name, shape in param_shapes.items():
            if len(shape) <= 1:
                # Parâmetros pequenos não são shardados
                rules[name] = None
            elif "model" in mesh_axes and len(shape) >= 2:
                # Shardar matrizes grandes no eixo do modelo
                rules[name] = "model"
            elif "batch" in mesh_axes:
                # Shardar no eixo do batch como fallback
                rules[name] = "batch"
            else:
                rules[name] = None
        return rules
