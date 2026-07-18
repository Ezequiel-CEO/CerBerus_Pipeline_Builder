#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPUManager: Gerenciador de operações de baixo nível com GPU para o CerBerus FMK.

Implementa acesso direto à GPU via CUDA sem dependências externas,
com fallback automático para CPU quando necessário.
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

logger = get_logger("gpu_manager")


class GPUManager:
    """
    Gerenciador de operações de GPU e memória para o CerBerus FMK.

    Implementa operações de baixo nível para manipulação de GPU e memória,
    com fallback automático para CPU quando a GPU não está disponível
    ou ocorre um erro.
    """

    def __init__(self, force_cpu: bool = False):
        """
        Inicializa o gerenciador de GPU.

        Args:
            force_cpu: Se True, força o uso de CPU mesmo que GPU esteja disponível
        """
        self.logger = logger
        self.force_cpu = force_cpu
        self.use_gpu = False
        self.cerberus_cupy = None

        # Dicionário para mapear ponteiros a nomes para debug
        self.memory_map = {}

        # Estado da memória
        self.total_allocated_gpu = 0
        self.total_allocated_cpu = 0

        # Detectar GPU
        if not force_cpu:
            self._initialize_gpu()
        else:
            self.logger.info("Modo CPU forçado pelo usuário")

        # Registrar estado inicial
        self._log_memory_state()

    def _initialize_gpu(self):
        """Inicializa o subsistema de GPU."""
        try:
            # Tentar importar o módulo CerBerusCuPy interno
            from cerberus_api.accelerate.cuda_ops import CerBerusCuPy

            # Tentar inicializar com GPU
            self.cerberus_cupy = CerBerusCuPy(use_gpu=True)

            # Verificar se GPU está realmente disponível
            if self.cerberus_cupy.is_gpu_available():
                self.use_gpu = True
                gpu_info = self._get_gpu_info()
                if gpu_info:
                    self.logger.info(
                        f"GPU detectada: {gpu_info['name']} ({gpu_info['memory_mb']}MB)"
                    )
                else:
                    self.logger.info(
                        "GPU detectada, mas não foi possível obter informações detalhadas"
                    )
            else:
                self.logger.warning("GPU não disponível, usando CPU como fallback")
                self.cerberus_cupy = CerBerusCuPy(use_gpu=False)

        except (ImportError, Exception) as e:
            self.logger.warning(f"Erro ao inicializar GPU: {e}")
            self.logger.info("Usando CPU como fallback")

            # Tentar usar implementação básica via NumPy
            self.cerberus_cupy = None

        # Definir device para uso
        self.device = "gpu" if self.use_gpu else "cpu"
        self.logger.info(f"Dispositivo ativo: {self.device.upper()}")

    def _get_gpu_info(self) -> Dict[str, Any]:
        """
        Obtém informações sobre a GPU.

        Returns:
            Dicionário com informações da GPU ou None
        """
        if not self.use_gpu or not self.cerberus_cupy:
            return None

        try:
            info = {"name": "GPU", "memory_mb": 0, "free_memory_mb": 0}

            # Obter informações via CerBerusCuPy
            if hasattr(self.cerberus_cupy, "get_gpu_info"):
                gpu_info = self.cerberus_cupy.get_gpu_info()
                if gpu_info:
                    # Extrair informações relevantes
                    info["name"] = gpu_info.get("name", "Desconhecida")
                    info["memory_mb"] = gpu_info.get("memory_total", 0)
                    info["free_memory_mb"] = gpu_info.get("memory_free", 0)

            return info

        except Exception as e:
            self.logger.error(f"Erro ao obter informações da GPU: {e}")
            return None

    def get_memory_info(self) -> Dict[str, float]:
        """
        Obtém informações sobre o uso de memória.

        Returns:
            Dicionário com informações de memória
        """
        info = {
            "ram_used_mb": 0,
            "ram_total_mb": 0,
            "vram_used_mb": 0,
            "vram_total_mb": 0,
            "ram_percent": 0,
            "vram_percent": 0,
        }

        # Obter informações de RAM
        try:
            import psutil

            process = psutil.Process(os.getpid())
            ram_used = process.memory_info().rss / (1024 * 1024)  # em MB
            ram_total = psutil.virtual_memory().total / (1024 * 1024)  # em MB

            info["ram_used_mb"] = ram_used
            info["ram_total_mb"] = ram_total
            info["ram_percent"] = (ram_used / ram_total) * 100
        except ImportError:
            # Fallback para estimativa básica se psutil não estiver disponível
            info["ram_used_mb"] = self.total_allocated_cpu

        # Obter informações de VRAM (se GPU estiver disponível)
        if self.use_gpu and self.cerberus_cupy:
            try:
                gpu_info = self._get_gpu_info()
                if gpu_info:
                    info["vram_total_mb"] = gpu_info.get("memory_mb", 0)
                    info["vram_used_mb"] = self.total_allocated_gpu

                    # Calcular percentual se total > 0
                    if info["vram_total_mb"] > 0:
                        info["vram_percent"] = (
                            info["vram_used_mb"] / info["vram_total_mb"]
                        ) * 100
            except Exception as e:
                self.logger.error(f"Erro ao obter informações de VRAM: {e}")

        return info

    def _log_memory_state(self):
        """Registra o estado atual da memória nos logs."""
        mem_info = self.get_memory_info()

        self.logger.info(
            f"Uso de memória: RAM {mem_info['ram_used_mb']:.1f}MB / "
            f"{mem_info['ram_total_mb']:.1f}MB ({mem_info['ram_percent']:.1f}%), "
            f"VRAM {mem_info['vram_used_mb']:.1f}MB / "
            f"{mem_info['vram_total_mb']:.1f}MB ({mem_info['vram_percent']:.1f}%)"
        )

    def to_device(self, data: np.ndarray, stream=None) -> Any:
        """
        Transfere dados para o dispositivo atual (GPU ou CPU).

        Args:
            data: Array NumPy para transferir
            stream: Stream CUDA opcional

        Returns:
            Dados no dispositivo alvo
        """
        if not self.use_gpu or not self.cerberus_cupy:
            return data

        try:
            # Registrar tamanho para contabilidade
            data_size_bytes = data.nbytes
            data_size_mb = data_size_bytes / (1024 * 1024)

            # Verificar se há memória suficiente
            mem_info = self.get_memory_info()
            free_vram = mem_info["vram_total_mb"] - mem_info["vram_used_mb"]

            # Se não houver memória suficiente na GPU, usar CPU
            if data_size_mb > free_vram * 0.9:  # Deixar margem de 10%
                self.logger.warning(
                    f"Memória insuficiente na GPU para alocar {data_size_mb:.1f}MB. "
                    f"Disponível: {free_vram:.1f}MB. Usando CPU."
                )
                return data

            # Transferir para GPU
            result = self.cerberus_cupy.to_device(data, stream)

            # Registrar alocação
            self.total_allocated_gpu += data_size_mb

            return result

        except Exception as e:
            self.logger.warning(f"Erro ao transferir dados para GPU: {e}. Usando CPU.")
            return data

    def to_host(self, data: Any) -> np.ndarray:
        """
        Transfere dados do dispositivo para CPU (host).

        Args:
            data: Dados no dispositivo

        Returns:
            Array NumPy na CPU
        """
        if not self.use_gpu or not self.cerberus_cupy or not self.is_on_device(data):
            return data

        try:
            # Transferir para CPU
            result = self.cerberus_cupy.to_host(data)

            # Atualizar contador de memória (estimativa)
            if hasattr(data, "nbytes"):
                data_size_mb = data.nbytes / (1024 * 1024)
                self.total_allocated_gpu -= min(data_size_mb, self.total_allocated_gpu)

            return result

        except Exception as e:
            self.logger.warning(f"Erro ao transferir dados para CPU: {e}")

            # Tentar retornar os dados originais em caso de erro
            if hasattr(data, "get"):
                try:
                    return data.get()
                except:
                    pass

            return data

    def is_on_device(self, data: Any) -> bool:
        """
        Verifica se os dados estão no dispositivo (GPU).

        Args:
            data: Dados a verificar

        Returns:
            True se os dados estiverem na GPU
        """
        if not self.use_gpu or not self.cerberus_cupy:
            return False

        try:
            return self.cerberus_cupy.is_device_array(data)
        except Exception:
            return False

    def clear_cache(self):
        """Limpa caches de memória."""
        if self.use_gpu and self.cerberus_cupy:
            try:
                # Limpar cache da GPU
                if hasattr(self.cerberus_cupy, "clear_cache"):
                    self.cerberus_cupy.clear_cache()

                # Método alternativo
                if hasattr(self.cerberus_cupy, "mempool_free_all_blocks"):
                    self.cerberus_cupy.mempool_free_all_blocks()

                # Resetar contador
                self.total_allocated_gpu = 0
            except Exception as e:
                self.logger.warning(f"Erro ao limpar cache da GPU: {e}")

        # Forçar coleta de lixo
        gc.collect()

        # Registrar estado de memória
        self._log_memory_state()

    def matmul(self, a: Any, b: Any) -> Any:
        """
        Multiplicação de matrizes otimizada para o dispositivo atual.

        Args:
            a: Primeira matriz
            b: Segunda matriz

        Returns:
            Resultado da multiplicação
        """
        # Caso GPU
        if self.use_gpu and self.cerberus_cupy and not self.force_cpu:
            try:
                # Verificar se as matrizes já estão na GPU
                a_gpu = a if self.is_on_device(a) else self.to_device(a)
                b_gpu = b if self.is_on_device(b) else self.to_device(b)

                # Multiplicação na GPU
                result_gpu = self.cerberus_cupy.matmul(a_gpu, b_gpu)

                return result_gpu

            except Exception as e:
                self.logger.warning(f"Erro em matmul na GPU: {e}. Caindo para CPU.")

                # Converter para CPU se necessário
                a_cpu = a if not self.is_on_device(a) else self.to_host(a)
                b_cpu = b if not self.is_on_device(b) else self.to_host(b)

                # Multiplicação na CPU
                return np.matmul(a_cpu, b_cpu)

        # Caso CPU
        return np.matmul(a, b)

    def softmax(self, x: Any, axis: int = -1) -> Any:
        """
        Função softmax otimizada para o dispositivo atual.

        Args:
            x: Tensor de entrada
            axis: Eixo para aplicar softmax

        Returns:
            Tensor após aplicação de softmax
        """
        # Caso GPU
        if self.use_gpu and self.cerberus_cupy and not self.force_cpu:
            try:
                # Verificar se o tensor já está na GPU
                x_gpu = x if self.is_on_device(x) else self.to_device(x)

                # Softmax na GPU
                if hasattr(self.cerberus_cupy, "softmax"):
                    return self.cerberus_cupy.softmax(x_gpu, axis=axis)

                # Implementação manual de softmax
                x_max = self.cerberus_cupy.max(x_gpu, axis=axis, keepdims=True)
                x_exp = self.cerberus_cupy.exp(x_gpu - x_max)
                x_sum = self.cerberus_cupy.sum(x_exp, axis=axis, keepdims=True)
                return x_exp / x_sum

            except Exception as e:
                self.logger.warning(f"Erro em softmax na GPU: {e}. Caindo para CPU.")

                # Converter para CPU se necessário
                x_cpu = x if not self.is_on_device(x) else self.to_host(x)

                # Softmax na CPU
                x_max = np.max(x_cpu, axis=axis, keepdims=True)
                x_exp = np.exp(x_cpu - x_max)
                x_sum = np.sum(x_exp, axis=axis, keepdims=True)
                return x_exp / x_sum

        # Caso CPU
        x_max = np.max(x, axis=axis, keepdims=True)
        x_exp = np.exp(x - x_max)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        return x_exp / x_sum

    def batch_matmul(
        self, tensors_a: List[Any], tensors_b: List[Any], max_batch_size: int = None
    ) -> List[Any]:
        """
        Executa multiplicação de matrizes em lote, otimizando o uso de memória.

        Args:
            tensors_a: Lista de primeiras matrizes
            tensors_b: Lista de segundas matrizes
            max_batch_size: Tamanho máximo do lote

        Returns:
            Lista dos resultados da multiplicação
        """
        if len(tensors_a) != len(tensors_b):
            raise ValueError("Listas de tensores devem ter o mesmo tamanho")

        # Determinar tamanho do lote baseado na memória disponível
        if max_batch_size is None:
            mem_info = self.get_memory_info()
            if self.use_gpu:
                free_mem = mem_info["vram_total_mb"] - mem_info["vram_used_mb"]
                # Heurística para batch size
                estimated_tensor_size = 0
                if len(tensors_a) > 0 and hasattr(tensors_a[0], "nbytes"):
                    estimated_tensor_size = tensors_a[0].nbytes / (1024 * 1024)

                if estimated_tensor_size > 0:
                    # 30% da memória livre por batch, considerando dados e resultados
                    max_batch_size = max(
                        1, int((free_mem * 0.3) / (estimated_tensor_size * 3))
                    )
                else:
                    max_batch_size = 8  # Valor padrão conservador
            else:
                max_batch_size = 16  # Batch maior para CPU

        # Processar em lotes
        results = []
        batch_count = (len(tensors_a) + max_batch_size - 1) // max_batch_size

        self.logger.debug(
            f"Processando {len(tensors_a)} operações em {batch_count} lotes"
        )

        for i in range(batch_count):
            start_idx = i * max_batch_size
            end_idx = min((i + 1) * max_batch_size, len(tensors_a))

            batch_a = tensors_a[start_idx:end_idx]
            batch_b = tensors_b[start_idx:end_idx]

            # Processar cada par de matrizes no lote
            batch_results = []
            for a, b in zip(batch_a, batch_b):
                result = self.matmul(a, b)
                batch_results.append(result)

            results.extend(batch_results)

            # Limpar memória após cada lote
            if self.use_gpu and i < batch_count - 1:
                self.clear_cache()

        return results

    def optimize_batch_size(
        self, tensor_shape: Tuple[int, ...], operator: str = "matmul"
    ) -> int:
        """
        Determina o tamanho ótimo de lote para operações de tensor.

        Args:
            tensor_shape: Forma do tensor
            operator: Tipo de operação ("matmul", "conv", etc.)

        Returns:
            Tamanho de lote otimizado
        """
        # Estimativa de espaço necessário por tensor
        try:
            # Criar um tensor de exemplo para estimar tamanho
            sample = np.zeros(tensor_shape, dtype=np.float32)
            bytes_per_tensor = sample.nbytes
            mb_per_tensor = bytes_per_tensor / (1024 * 1024)

            # Obter informações de memória
            mem_info = self.get_memory_info()

            # Usar VRAM se disponível, senão RAM
            if self.use_gpu:
                free_mem = mem_info["vram_total_mb"] - mem_info["vram_used_mb"]
                # Heurística: multiplicação de matrizes precisa de 3x o espaço
                # (entradas + saídas + operações intermediárias)
                if operator == "matmul":
                    multiplier = 3.0
                else:
                    multiplier = 2.0

                # Usar no máximo 70% da memória livre para evitar OOM
                max_batch_size = int((free_mem * 0.7) / (mb_per_tensor * multiplier))
            else:
                # Em CPU, podemos usar batches maiores
                free_mem = mem_info["ram_total_mb"] * 0.5  # Usar no máximo 50% da RAM
                max_batch_size = int(free_mem / (mb_per_tensor * 1.5))

            # Limites de segurança
            max_batch_size = max(1, min(max_batch_size, 128))

            self.logger.info(
                f"Batch size otimizado: {max_batch_size} para tensores de "
                f"{mb_per_tensor:.2f}MB cada (op: {operator})"
            )

            return max_batch_size

        except Exception as e:
            self.logger.warning(
                f"Erro ao otimizar batch size: {e}. Usando valor padrão."
            )
            return 4  # Valor padrão conservador

    def should_use_streaming(
        self, total_data_size_mb: float, max_item_size_mb: float
    ) -> bool:
        """
        Determina se o processamento deve ser feito em streaming.

        Args:
            total_data_size_mb: Tamanho total dos dados em MB
            max_item_size_mb: Tamanho do maior item em MB

        Returns:
            True se o streaming deve ser usado
        """
        # Obter informações de memória
        mem_info = self.get_memory_info()

        # Tamanho mínimo absoluto para streaming
        min_streaming_size = 100  # MB

        # Verificar condições para streaming
        if total_data_size_mb > min_streaming_size:
            self.logger.info(
                f"Habilitando streaming devido ao tamanho total dos dados: {total_data_size_mb:.1f}MB"
            )
            return True

        # Se qualquer item for muito grande, usar streaming
        if max_item_size_mb > 10:  # MB
            self.logger.info(
                f"Habilitando streaming devido ao tamanho do maior item: {max_item_size_mb:.1f}MB"
            )
            return True

        # Verificar em relação ao total de memória disponível
        if self.use_gpu:
            available_memory = mem_info["vram_total_mb"]
            # Usar no máximo 30% da VRAM total
            memory_threshold = available_memory * 0.3
        else:
            available_memory = mem_info["ram_total_mb"]
            # Usar no máximo 50% da RAM total
            memory_threshold = available_memory * 0.5

        should_stream = total_data_size_mb > memory_threshold

        if should_stream:
            self.logger.info(
                f"Habilitando streaming: dados ({total_data_size_mb:.1f}MB) > "
                f"limite de memória ({memory_threshold:.1f}MB)"
            )

        return should_stream
