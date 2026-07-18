#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CerBerus Optimizer Pro
=====================

Sistema avançado de otimização com auto-tuning e múltiplos backends
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.cuda.amp as amp
from dataclasses import dataclass
import concurrent.futures
import psutil
from functools import lru_cache

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from cerberus_api.utils.logging_config import get_logger
from cerberus_api.pipeline_builder.utils.memory_manager import MemoryManager

logger = get_logger("optimizer_pro")


@dataclass
class AdvancedOptimizationConfig:
    """Configurações avançadas de otimização."""

    # Configurações básicas
    use_mixed_precision: bool = True
    use_kernel_fusion: bool = True
    use_tensor_cores: bool = True
    min_gpu_usage: float = 0.80
    min_cpu_usage: float = 0.65
    max_batch_size: int = 32
    num_worker_threads: int = os.cpu_count()

    # Configurações avançadas
    enable_auto_tuning: bool = True
    enable_dynamic_batching: bool = True
    enable_smart_caching: bool = True
    enable_kernel_profiling: bool = True
    enable_multi_gpu: bool = True
    enable_pipeline_parallel: bool = True

    # Thresholds de performance
    perf_threshold_ms: float = 10.0
    memory_threshold_mb: float = 1024
    cache_size_gb: float = 4.0

    # Configurações de backend
    preferred_backend: str = "auto"  # "cuda", "triton", "jax", "cpu"
    fallback_order: List[str] = None

    def __post_init__(self):
        if self.fallback_order is None:
            self.fallback_order = ["cuda", "triton", "jax", "cpu"]


class CerberusOptimizerPro:
    """Otimizador avançado do Cerberus com auto-tuning."""

    def __init__(
        self,
        config: Optional[AdvancedOptimizationConfig] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        self.config = config or AdvancedOptimizationConfig()
        self.memory_manager = memory_manager or MemoryManager()

        # Inicialização dos backends
        self._init_backends()

        # Cache inteligente com LRU
        self._kernel_cache = {}
        self._op_cache = lru_cache(maxsize=1000)(self._compute_operation)

        # Métricas de performance
        self._perf_history = []
        self._current_backend = self._select_optimal_backend()

        logger.info(
            f"CerberusOptimizerPro inicializado com backend: {self._current_backend}"
        )

    def _init_backends(self):
        """Inicializa e configura todos os backends disponíveis."""
        self.available_backends = {"cpu": True}

        # CUDA/PyTorch setup
        if torch.cuda.is_available():
            self.available_backends["cuda"] = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.scaler = amp.GradScaler(enabled=self.config.use_mixed_precision)

            # Multi-GPU setup
            if self.config.enable_multi_gpu:
                self.num_gpus = torch.cuda.device_count()
                self.gpu_properties = [
                    {
                        "name": torch.cuda.get_device_properties(i).name,
                        "total_memory": torch.cuda.get_device_properties(
                            i
                        ).total_memory,
                        "compute_capability": torch.cuda.get_device_properties(i).major,
                    }
                    for i in range(self.num_gpus)
                ]
        else:
            self.available_backends["cuda"] = False

        # Triton setup
        self.available_backends["triton"] = HAS_TRITON
        if HAS_TRITON:
            self._init_triton_kernels()

        # JAX setup
        self.available_backends["jax"] = HAS_JAX
        if HAS_JAX:
            jax.config.update("jax_enable_x64", True)

        # CuPy setup
        self.available_backends["cupy"] = HAS_CUPY
        if HAS_CUPY:
            self._init_cupy_kernels()

    @staticmethod
    def _get_optimal_block_size(n: int) -> int:
        """Calcula tamanho ótimo de bloco para kernels."""
        return min(max(32, (n + 255) // 256 * 256), 1024)

    def _init_triton_kernels(self):
        """Inicializa kernels Triton otimizados."""
        if not HAS_TRITON:
            return

        @triton.jit
        def matmul_kernel(
            a_ptr,
            b_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE)
            num_pid_in_group = pid // num_pid_m
            group_id = pid % num_pid_m

            offs_am = group_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            offs_bn = num_pid_in_group * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            offs_k = tl.arange(0, BLOCK_SIZE)

            a = tl.load(
                a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b = tl.load(
                b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            c = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
            c += tl.dot(a, b)

            c_ptr = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
            tl.store(c_ptr, c)

        self._triton_kernels = {"matmul": matmul_kernel}

    def _init_cupy_kernels(self):
        """Inicializa kernels CuPy otimizados."""
        if not HAS_CUPY:
            return

        self._cupy_kernels = {
            "elementwise_add": cp.ElementwiseKernel(
                "T x, T y", "T z", "z = x + y", "elementwise_add"
            ),
            "elementwise_mul": cp.ElementwiseKernel(
                "T x, T y", "T z", "z = x * y", "elementwise_mul"
            ),
        }

    def _select_optimal_backend(self) -> str:
        """Seleciona o backend mais adequado baseado nas condições atuais."""
        if self.config.preferred_backend != "auto":
            if self.available_backends[self.config.preferred_backend]:
                return self.config.preferred_backend

        # Análise de hardware e carga
        gpu_memory = (
            self.memory_manager.get_memory_usage()[1]["free"]
            if torch.cuda.is_available()
            else 0
        )
        cpu_memory = psutil.virtual_memory().available

        # Decisão baseada em heurísticas
        if (
            gpu_memory > self.config.memory_threshold_mb * 1024 * 1024
        ):  # Converter para bytes
            if self.available_backends["cuda"]:
                return "cuda"
            elif self.available_backends["triton"]:
                return "triton"

        if (
            self.available_backends["jax"]
            and cpu_memory > self.config.memory_threshold_mb * 1024 * 1024
        ):
            return "jax"

        return "cpu"

    @torch.cuda.amp.autocast()
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Otimiza um modelo com técnicas avançadas."""
        if not torch.cuda.is_available():
            return model

        try:
            # Auto-tuning de hiperparâmetros
            if self.config.enable_auto_tuning:
                model = self._auto_tune_model(model)

            # Otimizações de kernel
            if self.config.use_kernel_fusion:
                model = self._fuse_operations(model)

            # Otimizações de pipeline
            if self.config.enable_pipeline_parallel and self.num_gpus > 1:
                model = self._pipeline_parallelize(model)

            # JIT compilation
            model = torch.jit.script(model)

            return model
        except Exception as e:
            logger.warning(f"Erro na otimização do modelo: {e}")
            return model

    def _auto_tune_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Auto-tuning de parâmetros do modelo."""
        if not self.config.enable_auto_tuning:
            return model

        # Análise de layers
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                # Otimizar parâmetros da layer
                optimal_params = self._find_optimal_params(module)
                self._apply_optimal_params(module, optimal_params)

        return model

    def _find_optimal_params(self, module: torch.nn.Module) -> Dict[str, Any]:
        """Encontra parâmetros ótimos para uma layer."""
        params = {}

        if isinstance(module, torch.nn.Linear):
            # Otimizar dimensões para Tensor Cores
            in_features = ((module.in_features + 127) // 128) * 128
            out_features = ((module.out_features + 127) // 128) * 128
            params.update({"in_features": in_features, "out_features": out_features})

        elif isinstance(module, torch.nn.Conv2d):
            # Otimizar parâmetros de convolução
            params.update(
                {"groups": self._optimize_conv_groups(module), "padding_mode": "zeros"}
            )

        return params

    def _optimize_conv_groups(self, conv_module: torch.nn.Conv2d) -> int:
        """Otimiza número de grupos para convoluções."""
        in_channels = conv_module.in_channels
        out_channels = conv_module.out_channels

        # Encontrar divisor comum que maximiza paralelismo
        possible_groups = [
            i
            for i in range(1, min(in_channels, out_channels) + 1)
            if in_channels % i == 0 and out_channels % i == 0
        ]

        return max(possible_groups)

    def _pipeline_parallelize(self, model: torch.nn.Module) -> torch.nn.Module:
        """Implementa paralelismo de pipeline para multi-GPU."""
        if not (self.config.enable_pipeline_parallel and self.num_gpus > 1):
            return model

        try:
            from torch.distributed.pipeline.sync import Pipe

            # Dividir modelo em chunks
            chunks = self._split_model(model)

            # Criar pipeline
            model = Pipe(chunks, chunks=self.num_gpus, checkpoint="never")

            return model
        except Exception as e:
            logger.warning(f"Erro ao paralelizar pipeline: {e}")
            return model

    def _split_model(self, model: torch.nn.Module) -> torch.nn.Sequential:
        """Divide modelo em chunks para pipeline parallelism."""
        layers = []
        current_device = 0

        for name, module in model.named_children():
            module.to(f"cuda:{current_device}")
            layers.append(module)

            # Alternar entre GPUs
            current_device = (current_device + 1) % self.num_gpus

        return torch.nn.Sequential(*layers)

    @torch.cuda.amp.autocast()
    def forward_pass(
        self,
        model: torch.nn.Module,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass otimizado com mixed precision e multi-GPU."""
        try:
            if isinstance(inputs, dict):
                return {k: self._forward_single(model, v) for k, v in inputs.items()}
            return self._forward_single(model, inputs)
        except Exception as e:
            logger.error(f"Erro no forward pass: {e}")
            raise

    def _forward_single(
        self, model: torch.nn.Module, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Processa um único forward pass."""
        if self.config.enable_dynamic_batching:
            batch_size = self.optimize_batch_size(inputs.shape)
            if inputs.shape[0] > batch_size:
                # Processar em mini-batches
                return self._process_mini_batches(model, inputs, batch_size)

        return model(inputs)

    def _process_mini_batches(
        self, model: torch.nn.Module, inputs: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Processa inputs em mini-batches."""
        outputs = []
        num_batches = (inputs.shape[0] + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, inputs.shape[0])
            batch_input = inputs[start_idx:end_idx]

            with torch.cuda.amp.autocast():
                batch_output = model(batch_input)
                outputs.append(batch_output)

        return torch.cat(outputs, dim=0)

    def optimize_memory(self) -> None:
        """Otimiza uso de memória GPU/CPU."""
        if torch.cuda.is_available():
            # Limpar cache PyTorch
            torch.cuda.empty_cache()

            # Otimizar alocação
            torch.cuda.memory.memory_allocated()
            torch.cuda.memory.max_memory_allocated()

            if HAS_CUPY:
                # Limpar cache CuPy
                cp.get_default_memory_pool().free_all_blocks()

        # Otimizar CPU
        import gc

        gc.collect()

    def get_performance_metrics(self) -> Dict[str, float]:
        """Obtém métricas detalhadas de performance."""
        metrics = super().get_performance_metrics()

        # Métricas adicionais
        if torch.cuda.is_available():
            metrics.update(
                {
                    "memory_allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
                    "max_memory_allocated": torch.cuda.max_memory_allocated() / 1024**2,
                    "current_stream": str(torch.cuda.current_stream()),
                    "kernel_time_avg": (
                        np.mean(self._perf_history) if self._perf_history else 0
                    ),
                }
            )

        # Métricas de CPU
        metrics.update(
            {
                "cpu_freq": psutil.cpu_freq().current,
                "cpu_temp": self._get_cpu_temp(),
                "ram_available": psutil.virtual_memory().available / 1024**2,
            }
        )

        return metrics

    @staticmethod
    def _get_cpu_temp() -> Optional[float]:
        """Obtém temperatura da CPU se disponível."""
        try:
            import psutil

            temps = psutil.sensors_temperatures()
            if temps and "coretemp" in temps:
                return sum(t.current for t in temps["coretemp"]) / len(
                    temps["coretemp"]
                )
        except:
            pass
        return None

    def __del__(self):
        """Cleanup ao destruir o otimizador."""
        self.optimize_memory()
        if hasattr(self, "_kernel_cache"):
            self._kernel_cache.clear()
        if hasattr(self, "_op_cache"):
            self._op_cache.cache_clear()
