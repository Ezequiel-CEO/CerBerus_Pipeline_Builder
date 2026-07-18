#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Abstração de Hardware para CerBerusFMK.

Este módulo fornece uma camada de abstração de hardware para o CerBerusFMK,
permitindo acesso direto a recursos de GPU/CPU sem dependências externas.
Implementa detecção automática de hardware, gerenciamento de recursos de memória
e interfaces nativas para operações de GPU.
"""

import os
import gc
import re
import sys
import json
import ctypes
import platform
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Configuração de logging
from cerberus_api.utils.logging_config import get_logger

logger = get_logger("hardware_abstraction")


class HardwareAbstraction:
    """
    Camada de abstração de hardware para o CerBerusFMK.

    Fornece acesso direto a recursos de hardware (CPU/GPU) usando
    implementações nativas, sem dependências de bibliotecas externas.
    """

    def __init__(self):
        """Inicializa a camada de abstração de hardware."""
        self.logger = logger

        # Informações do sistema
        self.system = self._detect_system()

        # Informações de CPU
        self.cpu_info = self._detect_cpu()

        # Informações de GPU
        self.gpu_info = self._detect_gpu()

        # Disponibilidade de CUDA
        self.cuda_available = self.gpu_info.get("cuda_available", False)

        # Bibliotecas nativas carregadas
        self.native_libs = {}

        # Registrar configuração de hardware
        self._log_hardware_config()

    def _detect_system(self) -> Dict[str, Any]:
        """
        Detecta informações do sistema operacional.

        Returns:
            Dicionário com informações do sistema
        """
        system_info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "cores_count": os.cpu_count(),
        }

        return system_info

    def _detect_cpu(self) -> Dict[str, Any]:
        """
        Detecta informações da CPU.

        Returns:
            Dicionário com informações da CPU
        """
        cpu_info = {
            "cores": os.cpu_count(),
            "architecture": platform.machine(),
            "supports_avx": False,
            "supports_avx2": False,
            "supports_avx512": False,
            "supports_vnni": False,
            "vendor": "Unknown",
        }

        try:
            # No Linux, usar /proc/cpuinfo
            if self.system["os"] == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()

                # Extrair fabricante
                if "GenuineIntel" in cpuinfo:
                    cpu_info["vendor"] = "Intel"
                elif "AuthenticAMD" in cpuinfo:
                    cpu_info["vendor"] = "AMD"

                # Detectar extensões
                cpu_info["supports_avx"] = "avx" in cpuinfo.lower()
                cpu_info["supports_avx2"] = "avx2" in cpuinfo.lower()
                cpu_info["supports_avx512"] = "avx512" in cpuinfo.lower()
                cpu_info["supports_vnni"] = "vnni" in cpuinfo.lower()

            # No Windows, usar outras técnicas
            elif self.system["os"] == "Windows":
                # Implementação para Windows pode ser adicionada posteriormente
                pass

        except Exception as e:
            self.logger.warning(f"Erro ao detectar CPU: {e}")

        return cpu_info

    def _detect_gpu(self) -> Dict[str, Any]:
        """
        Detecta informações da GPU.

        Returns:
            Dicionário com informações da GPU
        """
        gpu_info = {
            "has_gpu": False,
            "cuda_available": False,
            "gpus": [],
            "cuda_version": None,
            "driver_version": None,
        }

        try:
            # Verificar nvidia-smi no Linux
            if self.system["os"] == "Linux":
                try:
                    # Executar nvidia-smi
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=name,memory.total,memory.free,compute_cap",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=3,
                    )

                    if result.returncode == 0:
                        lines = result.stdout.strip().split("\n")
                        gpus = []

                        for i, line in enumerate(lines):
                            parts = [part.strip() for part in line.split(",")]

                            # Verificar se temos formato esperado
                            if len(parts) >= 3:
                                gpu = {
                                    "id": i,
                                    "name": parts[0],
                                    "memory_total_mb": float(parts[1]),
                                    "memory_free_mb": float(parts[2]),
                                    "compute_capability": (
                                        parts[3] if len(parts) > 3 else "Unknown"
                                    ),
                                }

                                gpus.append(gpu)

                        if gpus:
                            gpu_info["has_gpu"] = True
                            gpu_info["cuda_available"] = True
                            gpu_info["gpus"] = gpus

                            # Tentar obter versão CUDA
                            cuda_result = subprocess.run(
                                [
                                    "nvidia-smi",
                                    "--query-gpu=driver_version,cuda_version",
                                    "--format=csv,noheader,nounits",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=3,
                            )

                            if cuda_result.returncode == 0:
                                cuda_lines = cuda_result.stdout.strip().split("\n")
                                if cuda_lines:
                                    parts = [
                                        part.strip()
                                        for part in cuda_lines[0].split(",")
                                    ]
                                    if len(parts) >= 2:
                                        gpu_info["driver_version"] = parts[0]
                                        gpu_info["cuda_version"] = parts[1]
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass

        except Exception as e:
            self.logger.warning(f"Erro ao detectar GPU: {e}")

        return gpu_info

    def _log_hardware_config(self):
        """Registra configuração de hardware detectada."""
        if self.gpu_info["has_gpu"]:
            gpu_names = [gpu["name"] for gpu in self.gpu_info["gpus"]]
            self.logger.info(f"GPU(s) detectada(s): {', '.join(gpu_names)}")
            self.logger.info(
                f"CUDA disponível: {self.cuda_available} (versão: {self.gpu_info['cuda_version']})"
            )
        else:
            self.logger.info("Nenhuma GPU detectada, usando apenas CPU")

        self.logger.info(
            f"CPU: {self.cpu_info['vendor']} com {self.cpu_info['cores']} núcleos"
        )

        if self.cpu_info["supports_avx2"]:
            self.logger.info("CPU suporta AVX2 para aceleração")

    def get_memory_info(self) -> Dict[str, float]:
        """
        Obtém informações de memória do sistema.

        Returns:
            Dicionário com uso de RAM e VRAM
        """
        memory_info = {
            "ram_total_mb": 0,
            "ram_free_mb": 0,
            "ram_used_mb": 0,
            "vram_total_mb": 0,
            "vram_free_mb": 0,
            "vram_used_mb": 0,
        }

        try:
            # Obter informações de RAM
            if self.system["os"] == "Linux":
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()

                # Extrair valores em kB e converter para MB
                match = re.search(r"MemTotal:\s+(\d+)", meminfo)
                if match:
                    memory_info["ram_total_mb"] = int(match.group(1)) / 1024

                match = re.search(r"MemFree:\s+(\d+)", meminfo)
                if match:
                    memory_info["ram_free_mb"] = int(match.group(1)) / 1024

                match = re.search(r"MemAvailable:\s+(\d+)", meminfo)
                if match:
                    memory_info["ram_free_mb"] = int(match.group(1)) / 1024

                # Calcular uso
                memory_info["ram_used_mb"] = (
                    memory_info["ram_total_mb"] - memory_info["ram_free_mb"]
                )

            # Obter informações de VRAM (se disponível)
            if self.gpu_info["has_gpu"]:
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.total,memory.free,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=3,
                    )

                    if result.returncode == 0:
                        lines = result.stdout.strip().split("\n")
                        if lines:
                            parts = [
                                float(part.strip()) for part in lines[0].split(",")
                            ]
                            if len(parts) >= 3:
                                memory_info["vram_total_mb"] = parts[0]
                                memory_info["vram_free_mb"] = parts[1]
                                memory_info["vram_used_mb"] = parts[2]
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass

        except Exception as e:
            self.logger.warning(f"Erro ao obter informações de memória: {e}")

        return memory_info

    def is_cuda_available(self) -> bool:
        """
        Verifica se CUDA está disponível no sistema.

        Returns:
            True se CUDA estiver disponível
        """
        return self.cuda_available

    def get_optimal_device(self) -> str:
        """
        Determina o dispositivo ótimo para computação.

        Returns:
            'gpu' ou 'cpu'
        """
        if self.gpu_info["has_gpu"] and self.cuda_available:
            return "gpu"
        return "cpu"

    def should_use_streaming(self, data_size_mb: float) -> bool:
        """
        Determina se processamento em streaming deve ser usado.

        Args:
            data_size_mb: Tamanho dos dados em MB

        Returns:
            True se streaming for recomendado
        """
        mem_info = self.get_memory_info()

        # Usar streaming se dados excederem 30% da RAM disponível
        if data_size_mb > 0.3 * mem_info["ram_free_mb"]:
            return True

        # Ou se for um volume muito grande de dados
        if data_size_mb > 1000:  # 1GB
            return True

        return False

    def optimize_batch_size(self, item_size_mb: float, device: str = "auto") -> int:
        """
        Calcula o tamanho de batch ótimo baseado no hardware.

        Args:
            item_size_mb: Tamanho de cada item em MB
            device: Dispositivo a usar ('gpu', 'cpu', ou 'auto')

        Returns:
            Tamanho de batch recomendado
        """
        if device == "auto":
            device = self.get_optimal_device()

        mem_info = self.get_memory_info()

        if device == "gpu" and self.gpu_info["has_gpu"]:
            # Usar no máximo 70% da memória livre da GPU
            free_vram = mem_info["vram_free_mb"]
            max_batch_mb = free_vram * 0.7

            # Tamanho de batch ótimo
            max_batch = max(1, int(max_batch_mb / item_size_mb))

            # Limitar a 32 para eficiência
            return min(32, max_batch)
        else:
            # Usar no máximo 50% da RAM livre para batch
            free_ram = mem_info["ram_free_mb"]
            max_batch_mb = free_ram * 0.5

            # Tamanho de batch ótimo
            max_batch = max(1, int(max_batch_mb / item_size_mb))

            # Limitar a 16 para evitar sobrecarga
            return min(16, max_batch)

    def clear_memory_cache(self):
        """Limpa caches e libera memória não utilizada."""
        # Forçar coleta de lixo
        gc.collect()

        # No Linux, tentar liberar caches
        if self.system["os"] == "Linux":
            try:
                # Limpar caches de arquivo no Linux
                with open("/proc/sys/vm/drop_caches", "w") as f:
                    f.write("1")
            except (PermissionError, IOError):
                # Silenciosamente ignorar se não tivermos permissão
                pass


# Instância global para uso em todo o framework
hardware = HardwareAbstraction()
