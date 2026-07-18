#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gerenciador de Memória
=====================

Monitora e gerencia uso de memória RAM e VRAM.
"""

import os
import psutil
import logging
import gc
from typing import Dict, Any, Optional, Tuple, Union, List
import time
import torch
import numpy as np

# Tentar importar CUDA
try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

from cerberus_api.utils.logging_config import get_logger

logger = get_logger("memory_manager")


class MemoryManager:
    """Gerencia memória RAM e VRAM."""

    def __init__(
        self,
        ram_threshold: float = 0.85,  # 85% de uso de RAM dispara limpeza
        vram_threshold: float = 0.80,  # 80% de uso de VRAM dispara offloading
        enable_streaming: bool = True,  # Habilitar processamento em streaming
        max_batch_size: int = 16,  # Tamanho máximo de lote para processamento
        log_interval: int = 10,  # Intervalo de log de memória (segundos)
    ):
        """
        Inicializa o gerenciador de memória.

        Args:
            ram_threshold: Limiar de uso de RAM que dispara limpeza automática
            vram_threshold: Limiar de uso de VRAM que dispara offloading
            enable_streaming: Se deve habilitar processamento em streaming para grandes volumes
            max_batch_size: Tamanho máximo de lote para processamento em batch
            log_interval: Intervalo em segundos para registrar uso de memória
        """
        self.ram_threshold = ram_threshold
        self.vram_threshold = vram_threshold
        self.enable_streaming = enable_streaming
        self.max_batch_size = max_batch_size
        self.log_interval = log_interval

        self.process = psutil.Process(os.getpid())
        self.cuda_available = CUDA_AVAILABLE

        if self.cuda_available:
            self.device = torch.device("cuda")
            self.gpu_name = torch.cuda.get_device_name()
            self.total_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(
                f"CUDA disponível. GPUs: {torch.cuda.device_count()}, Modelo: {self.gpu_name}"
            )
        else:
            self.device = None
            logger.warning("CUDA não disponível. Usando apenas CPU.")

        # Registrar uso inicial
        ram, vram = self.get_memory_usage()
        logger.info(
            f"Uso de memória inicialização: "
            f"RAM {ram['used']:.1f}MB / {ram['total']:.1f}MB ({ram['percent']:.1f}%), "
            f"VRAM {vram['used']:.1f}MB / {vram['total']:.1f}MB ({vram['percent']:.1f}%)"
        )

        # Modelos carregados
        self.loaded_models = {}

        # Último log de memória
        self.last_log_time = 0

        # Cache de embeddings
        self.embedding_cache = {}
        self.max_cache_size = 1000  # Número máximo de entradas no cache

        # Cache de tensores
        self.tensor_cache = {}
        self.cached_memory = 0

        # Monitoramento
        self._log_initial_memory()

    def _log_initial_memory(self):
        """Registra uso inicial de memória."""
        if self.cuda_available:
            ram_total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            ram_used = ram_total - (
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
            )

            vram_total = self.total_memory / (1024**2)  # MB
            vram_used = torch.cuda.memory_allocated() / (1024**2)  # MB

            logger.info(
                f"Uso de memória inicialização: "
                f"RAM {ram_used/1024**2:.1f}MB / {ram_total/1024**2:.1f}MB ({ram_used/ram_total*100:.1f}%), "
                f"VRAM {vram_used:.1f}MB / {vram_total:.1f}MB ({vram_used/vram_total*100:.1f}%)"
            )

    def get_memory_usage(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Obtém uso atual de memória RAM e VRAM.

        Returns:
            Tuple[Dict, Dict]: Dicionários com informações de RAM e VRAM
        """
        # RAM
        ram_info = psutil.virtual_memory()
        ram = {
            "total": ram_info.total / (1024 * 1024),  # MB
            "used": ram_info.used / (1024 * 1024),  # MB
            "free": ram_info.free / (1024 * 1024),  # MB
            "percent": ram_info.percent,
        }

        # VRAM
        if self.cuda_available:
            vram_total = self.total_memory / (1024 * 1024)  # MB
            vram_used = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            vram_free = vram_total - vram_used
            vram_percent = (vram_used / vram_total) * 100 if vram_total > 0 else 0

            vram = {
                "total": vram_total,
                "used": vram_used,
                "free": vram_free,
                "percent": vram_percent,
            }
        else:
            vram = {"total": 0.0, "used": 0.0, "free": 0.0, "percent": 0.0}

        return ram, vram

    def get_gpu_memory_usage(self) -> float:
        """
        Retorna uso de memória GPU em MB.

        Returns:
            float: Memória GPU usada em MB
        """
        if self.cuda_available:
            return torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        return 0.0

    def clear_gpu_cache(self):
        """Limpa cache da GPU."""
        if self.cuda_available:
            torch.cuda.empty_cache()

    def check_memory_available(
        self, required_ram: float, required_vram: Optional[float] = None
    ) -> bool:
        """
        Verifica se há memória suficiente disponível.

        Args:
            required_ram: RAM necessária em MB
            required_vram: VRAM necessária em MB (opcional)

        Returns:
            bool: True se há memória suficiente
        """
        ram, vram = self.get_memory_usage()

        # Verificar RAM
        if required_ram > ram["free"]:
            return False

        # Verificar VRAM se necessário
        if required_vram and self.cuda_available:
            if required_vram > vram["free"]:
                return False

        return True

    def log_memory_usage(self, context: str = "") -> Dict[str, float]:
        """
        Registra uso atual de memória, respeitando intervalo mínimo de log.

        Args:
            context: Informação contextual para o log

        Returns:
            Informações de uso de memória
        """
        current_time = time.time()

        # Verificar intervalo de log
        if current_time - self.last_log_time < self.log_interval:
            return {}

        self.last_log_time = current_time

        memory_info = self.get_memory_usage()

        log_message = (
            f"Uso de memória {context}: RAM {memory_info[0]['used']:.1f}MB / "
            + f"{memory_info[0]['total']:.1f}MB ({memory_info[0]['percent'] * 100:.1f}%)"
        )

        if self.cuda_available and "used" in memory_info[1]:
            log_message += (
                f", VRAM {memory_info[1]['used']:.1f}MB / "
                + f"{memory_info[1]['total']:.1f}MB ({memory_info[1]['percent'] * 100:.1f}%)"
            )

        logger.info(log_message)

        # Verificar se é necessário limpeza de memória
        self._check_memory_thresholds(memory_info)

        return memory_info[0]

    def _check_memory_thresholds(
        self, memory_info: Tuple[Dict[str, float], Dict[str, float]]
    ) -> None:
        """
        Verifica se limites de memória foram atingidos e toma ações necessárias.

        Args:
            memory_info: Informações de uso de memória
        """
        # Verificar RAM
        ram_percent = memory_info[0]["percent"]
        if ram_percent > self.ram_threshold:
            logger.warning(
                f"Uso de RAM ({ram_percent * 100:.1f}%) acima do limite ({self.ram_threshold * 100:.1f}%). Limpando memória..."
            )
            self.clear_memory()

        # Verificar VRAM
        vram_percent = memory_info[1]["percent"]
        if self.cuda_available and vram_percent > self.vram_threshold:
            logger.warning(
                f"Uso de VRAM ({vram_percent * 100:.1f}%) acima do limite ({self.vram_threshold * 100:.1f}%). Realizando offload de modelos..."
            )
            self.offload_models_to_cpu()

    def clear_memory(self) -> None:
        """
        Libera memória não utilizada.
        """
        # Limpar cache de embeddings
        self.clear_embedding_cache()

        # Coletar lixo do Python
        gc.collect()

        # Limpar cache CUDA se disponível
        if self.cuda_available:
            try:
                torch.cuda.empty_cache()
                logger.info("Cache CUDA limpo")
            except Exception as e:
                logger.error(f"Erro ao limpar cache CUDA: {e}")

        logger.info("Limpeza de memória concluída")

    def clear_embedding_cache(self) -> None:
        """
        Limpa o cache de embeddings.
        """
        cache_size = len(self.embedding_cache)
        if cache_size > 0:
            logger.info(f"Limpando cache de embeddings ({cache_size} entradas)")
            self.embedding_cache.clear()

    def load_model(self, model_name: str, model_class: Any, **kwargs) -> Any:
        """
        Carrega um modelo, verifica uso de memória e realiza offload se necessário.

        Args:
            model_name: Nome do modelo para identificação
            model_class: Classe do modelo a ser carregado
            **kwargs: Parâmetros adicionais para inicialização do modelo

        Returns:
            Instância do modelo carregado
        """
        # Verificar se o modelo já está carregado
        if model_name in self.loaded_models:
            logger.info(f"Usando modelo '{model_name}' já carregado")
            return self.loaded_models[model_name]

        # Verificar uso de memória antes de carregar
        self.log_memory_usage(f"antes de carregar modelo '{model_name}'")

        try:
            # Verificar se deve usar GPU
            if (
                self.cuda_available
                and "device" not in kwargs
                and hasattr(model_class, "to")
            ):
                kwargs["device"] = self.device

            # Carregar modelo
            logger.info(f"Carregando modelo '{model_name}'...")
            model = model_class(**kwargs)

            # Mover para GPU se possível e necessário
            if (
                self.cuda_available
                and hasattr(model, "to")
                and not kwargs.get("device")
            ):
                model = model.to(self.device)
                logger.info(f"Modelo '{model_name}' movido para {self.device}")

            # Registrar modelo
            self.loaded_models[model_name] = model

            # Verificar uso de memória após carregar
            self.log_memory_usage(f"após carregar modelo '{model_name}'")

            return model

        except Exception as e:
            logger.error(f"Erro ao carregar modelo '{model_name}': {e}")
            raise

    def offload_models_to_cpu(self) -> None:
        """
        Move modelos da GPU para CPU para liberar VRAM.
        """
        if not self.cuda_available:
            return

        moved_count = 0
        for name, model in self.loaded_models.items():
            try:
                if (
                    hasattr(model, "to")
                    and hasattr(model, "device")
                    and str(model.device) != "cpu"
                ):
                    logger.info(f"Movendo modelo '{name}' para CPU")
                    model.to("cpu")
                    moved_count += 1
            except Exception as e:
                logger.error(f"Erro ao mover modelo '{name}' para CPU: {e}")

        if moved_count > 0:
            # Limpar cache CUDA
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            logger.info(f"{moved_count} modelos movidos para CPU")

    def optimize_batch_size(self, data_size: int) -> int:
        """
        Calcula tamanho de lote ideal com base no tamanho dos dados e memória disponível.

        Args:
            data_size: Tamanho total do conjunto de dados

        Returns:
            Tamanho de lote otimizado
        """
        memory_info = self.get_memory_usage()

        # Tamanho de lote base
        batch_size = self.max_batch_size

        # Reduzir tamanho se memória estiver ocupada
        ram_percent = memory_info[0]["percent"]
        if ram_percent > 0.7:  # Se RAM > 70%
            reduction_factor = 1.0 - (
                (ram_percent - 0.7) * 2
            )  # Reduzir proporcionalmente
            batch_size = max(1, int(batch_size * reduction_factor))

        # Reduzir mais se VRAM estiver ocupada
        if self.cuda_available:
            vram_percent = memory_info[1]["percent"]
            if vram_percent > 0.6:  # Se VRAM > 60%
                reduction_factor = 1.0 - ((vram_percent - 0.6) * 2)
                batch_size = max(1, int(batch_size * reduction_factor))

        # Considerar tamanho do conjunto de dados
        batch_size = min(
            batch_size, max(1, data_size // 10)
        )  # No máximo 1/10 dos dados

        logger.info(f"Tamanho de lote otimizado: {batch_size} para {data_size} itens")

        return batch_size

    def should_process_in_streaming(
        self, total_data_size: int, item_sizes: List[int]
    ) -> bool:
        """
        Determina se os dados devem ser processados em modo streaming.

        Args:
            total_data_size: Tamanho total dos dados em bytes
            item_sizes: Lista de tamanhos de cada item em bytes

        Returns:
            True se processar em streaming, False caso contrário
        """
        if not self.enable_streaming:
            return False

        # Verificar tamanho total - forçar streaming para dados acima de 100MB
        if total_data_size > 100 * 1024 * 1024:
            logger.info(
                f"Habilitando processamento em streaming - Dados grandes: {total_data_size/(1024*1024):.1f}MB"
            )
            return True

        # Verificar se há itens muito grandes (mais de 10MB)
        max_item_size = max(item_sizes) if item_sizes else 0
        if max_item_size > 10 * 1024 * 1024:
            logger.info(
                f"Habilitando processamento em streaming - Item grande: {max_item_size/(1024*1024):.1f}MB"
            )
            return True

        # Verificar tamanho total em relação à RAM disponível
        memory_info = self.get_memory_usage()
        ram_available_mb = memory_info[0]["total"] * (1 - memory_info[0]["percent"])

        # Converter para MB para comparação
        total_data_mb = total_data_size / (1024 * 1024)

        # Se dados são grandes em relação à RAM disponível (acima de 30% da RAM livre)
        if total_data_mb > ram_available_mb * 0.3:
            logger.info(
                f"Habilitando processamento em streaming - Dados: {total_data_mb:.1f}MB, RAM disponível: {ram_available_mb:.1f}MB"
            )
            return True

        return False

    def cache_embedding(self, text_id: str, embedding: Any) -> None:
        """
        Armazena um embedding no cache.

        Args:
            text_id: Identificador único do texto
            embedding: Vetor de embedding
        """
        # Limpar cache se atingir tamanho máximo
        if len(self.embedding_cache) >= self.max_cache_size:
            # Remover metade das entradas mais antigas
            keys_to_remove = list(self.embedding_cache.keys())[
                : self.max_cache_size // 2
            ]
            for key in keys_to_remove:
                del self.embedding_cache[key]

            logger.info(
                f"Cache de embeddings reduzido: {len(keys_to_remove)} itens removidos"
            )

        # Armazenar no cache
        self.embedding_cache[text_id] = embedding

    def get_cached_embedding(self, text_id: str) -> Optional[Any]:
        """
        Recupera um embedding do cache.

        Args:
            text_id: Identificador único do texto

        Returns:
            Embedding armazenado ou None se não encontrado
        """
        return self.embedding_cache.get(text_id)

    def allocate(self, shape: Tuple[int, ...], dtype) -> torch.Tensor:
        """
        Aloca um tensor na GPU.

        Args:
            shape: Formato do tensor
            dtype: Tipo de dados

        Returns:
            Tensor alocado
        """
        if not self.cuda_available:
            return torch.zeros(shape, dtype=dtype)

        # Calcular tamanho necessário
        size_bytes = np.prod(shape) * torch.tensor([], dtype=dtype).element_size()

        # Verificar cache
        cache_key = f"{shape}_{dtype}"
        if cache_key in self.tensor_cache:
            tensor = self.tensor_cache[cache_key]
            if tensor.shape == shape and tensor.dtype == dtype:
                return tensor

        # Verificar memória disponível
        available = self.total_memory - torch.cuda.memory_allocated()
        if size_bytes > available:
            self._free_memory(size_bytes - available)

        # Alocar novo tensor
        tensor = torch.zeros(shape, dtype=dtype, device=self.device)

        # Atualizar cache
        self.tensor_cache[cache_key] = tensor
        self.cached_memory += size_bytes

        return tensor

    def _free_memory(self, bytes_needed: int):
        """
        Libera memória para acomodar uma nova alocação.

        Args:
            bytes_needed: Bytes necessários
        """
        # Forçar coleta de lixo
        gc.collect()
        torch.cuda.empty_cache()

        # Se ainda precisar liberar mais
        if bytes_needed > 0:
            # Remover tensores do cache
            keys = list(self.tensor_cache.keys())
            for key in keys:
                tensor = self.tensor_cache[key]
                size = tensor.numel() * tensor.element_size()

                del self.tensor_cache[key]
                del tensor

                self.cached_memory -= size
                bytes_needed -= size

                if bytes_needed <= 0:
                    break

        # Forçar sincronização
        torch.cuda.synchronize()

    def get_memory_info(self) -> Dict[str, float]:
        """
        Retorna informações sobre uso de memória.

        Returns:
            Dict com informações de memória
        """
        if not self.cuda_available:
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "cached_mb": 0}

        total = self.total_memory / (1024**2)  # MB
        used = torch.cuda.memory_allocated() / (1024**2)  # MB
        cached = self.cached_memory / (1024**2)  # MB

        return {
            "total_mb": total,
            "used_mb": used,
            "free_mb": total - used,
            "cached_mb": cached,
        }

    def clear_cache(self):
        """Limpa o cache de tensores."""
        self.tensor_cache.clear()
        self.cached_memory = 0
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def __del__(self):
        """Cleanup ao destruir o objeto."""
        self.clear_cache()


# Instância global do gerenciador de memória
memory_manager = MemoryManager()
