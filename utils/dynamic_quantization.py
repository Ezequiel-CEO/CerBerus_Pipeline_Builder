#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantização Dinâmica
===================

Implementação otimizada de quantização dinâmica com suporte a:
- INT4/8/16/32/64
- Calibração automática
- Quantização por canal
- Dequantização otimizada
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from enum import Enum
import numpy as np


class QuantizationType(Enum):
    """Tipos de quantização suportados."""

    INT4 = 4
    INT8 = 8
    INT16 = 16
    INT32 = 32
    INT64 = 64


@dataclass
class QuantizationConfig:
    """Configurações para quantização dinâmica."""

    qtype: QuantizationType = QuantizationType.INT8
    per_channel: bool = True
    symmetric: bool = True
    calibration_samples: int = 100
    min_scale: float = 1e-5
    max_scale: float = 1e5
    num_bits: int = 8
    optimize_zero_point: bool = True


class DynamicQuantization(torch.nn.Module):
    """
    Implementação de quantização dinâmica com calibração automática.
    Suporta múltiplas precisões e otimizações por canal.
    """

    def __init__(self, config: QuantizationConfig):
        super().__init__()
        self.config = config

        # Calcular parâmetros de quantização
        self.qmin = -(2 ** (self.config.qtype.value - 1))
        self.qmax = (2 ** (self.config.qtype.value - 1)) - 1

        # Registrar buffers
        self.register_buffer("scale", None)
        self.register_buffer("zero_point", None)

    def _calculate_qparams(
        self, x: torch.Tensor, channel_dim: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula parâmetros de quantização.

        Args:
            x: Tensor de entrada
            channel_dim: Dimensão dos canais para quantização por canal

        Returns:
            Tuple com escala e zero point
        """
        if self.config.per_channel:
            dims = [i for i in range(x.dim()) if i != channel_dim]
            xmin = x.amin(dim=dims, keepdim=True)
            xmax = x.amax(dim=dims, keepdim=True)
        else:
            xmin = x.min()
            xmax = x.max()

        if self.config.symmetric:
            abs_max = torch.max(xmin.abs(), xmax.abs())
            xmin = -abs_max
            xmax = abs_max

        scale = (xmax - xmin) / (self.qmax - self.qmin)
        scale = torch.clamp(scale, self.config.min_scale, self.config.max_scale)

        if self.config.optimize_zero_point:
            zero_point = self.qmin - torch.round(xmin / scale)
            zero_point = torch.clamp(zero_point, self.qmin, self.qmax)
        else:
            zero_point = torch.zeros_like(scale)

        return scale, zero_point

    def _quantize(
        self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Quantiza o tensor de entrada.

        Args:
            x: Tensor de entrada
            scale: Escala de quantização
            zero_point: Zero point

        Returns:
            Tensor quantizado
        """
        x_scaled = x / scale
        x_scaled = torch.round(x_scaled)
        x_quant = torch.clamp(x_scaled + zero_point, self.qmin, self.qmax)
        return x_quant

    def _dequantize(
        self, x_quant: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantiza o tensor.

        Args:
            x_quant: Tensor quantizado
            scale: Escala de quantização
            zero_point: Zero point

        Returns:
            Tensor dequantizado
        """
        return scale * (x_quant - zero_point)

    def calibrate(self, x: torch.Tensor, channel_dim: int = -1):
        """
        Calibra os parâmetros de quantização.

        Args:
            x: Tensor de calibração
            channel_dim: Dimensão dos canais
        """
        with torch.no_grad():
            # Amostrar dados de calibração
            if x.shape[0] > self.config.calibration_samples:
                idx = torch.randperm(x.shape[0])[: self.config.calibration_samples]
                x = x[idx]

            # Calcular parâmetros
            scale, zero_point = self._calculate_qparams(x, channel_dim)

            # Registrar buffers
            self.register_buffer("scale", scale)
            self.register_buffer("zero_point", zero_point)

    def forward(
        self, x: torch.Tensor, channel_dim: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass do módulo.

        Args:
            x: Tensor de entrada
            channel_dim: Dimensão dos canais

        Returns:
            Tuple com tensor quantizado e dequantizado
        """
        # Calibrar se necessário
        if self.scale is None or self.zero_point is None:
            self.calibrate(x, channel_dim)

        # Quantizar
        x_quant = self._quantize(x, self.scale, self.zero_point)

        # Dequantizar
        x_dequant = self._dequantize(x_quant, self.scale, self.zero_point)

        return x_quant, x_dequant

    def extra_repr(self) -> str:
        """Representação string com parâmetros."""
        return (
            f"qtype={self.config.qtype.value}, "
            f"per_channel={self.config.per_channel}, "
            f"symmetric={self.config.symmetric}"
        )
