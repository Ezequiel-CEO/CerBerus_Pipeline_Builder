#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo avançado para carregamento e gerenciamento de modelos GGUF
"""

import os
import logging
import threading
import queue
import numpy as np
from typing import Optional, Dict, Any, Union, List, Generator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


@dataclass
class GGUFConfig:
    """Configuração avançada para modelos GGUF"""

    model_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    max_memory: Optional[Dict[int, str]] = None
    offload_folder: Optional[str] = None
    use_safetensors: bool = True
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True


class GGUFModel:
    """
    Implementação avançada para carregamento e inferência de modelos GGUF
    """

    def __init__(self, config: GGUFConfig):
        """
        Inicializa o carregador de modelo GGUF

        Args:
            config: Configuração do modelo
        """
        self.config = config
        self.model_path = Path(config.model_path)
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._inference_queue = queue.Queue()
        self._stop_event = threading.Event()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Arquivo do modelo não encontrado: {self.model_path}"
            )

        # Configurar logging
        self.logger = logging.getLogger(f"GGUFModel_{self.model_path.stem}")

    def _load_model(self) -> bool:
        """Carrega o modelo e tokenizer"""
        try:
            with self._lock:
                if self._model is not None:
                    return True

            self.logger.info(f"Carregando modelo GGUF: {self.model_path}")

            # Carregar tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), trust_remote_code=self.config.trust_remote_code
            )

            # Configurar carregamento do modelo
            model_kwargs = {
                "torch_dtype": self.config.dtype,
                "device_map": "auto",
                "trust_remote_code": self.config.trust_remote_code,
                "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            }

            if self.config.max_memory:
                model_kwargs["max_memory"] = self.config.max_memory

            if self.config.offload_folder:
                model_kwargs["offload_folder"] = self.config.offload_folder

            # Carregar modelo
            if (
                self.config.use_safetensors
                and (self.model_path / "model.safetensors").exists()
            ):
                self.logger.info("Usando safetensors para carregamento")
                state_dict = load_file(self.model_path / "model.safetensors")
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path), state_dict=state_dict, **model_kwargs
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path), **model_kwargs
                )

            self.logger.info("Modelo carregado com sucesso")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            raise

    def _unload_model(self) -> bool:
        """Descarrega o modelo e libera memória"""
        try:
            with self._lock:
                if self._model is not None:
                    del self._model
                    self._model = None
                    torch.cuda.empty_cache()

                if self._tokenizer is not None:
                    del self._tokenizer
                    self._tokenizer = None

                self.logger.info("Modelo descarregado com sucesso")
                return True

        except Exception as e:
            self.logger.error(f"Erro ao descarregar modelo: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações detalhadas do modelo"""
        if self._model is None:
            raise RuntimeError("Modelo não carregado")

        info = {
            "model_type": self._model.config.model_type,
            "vocab_size": self._model.config.vocab_size,
            "hidden_size": self._model.config.hidden_size,
            "num_attention_heads": self._model.config.num_attention_heads,
            "num_hidden_layers": self._model.config.num_hidden_layers,
            "max_position_embeddings": self._model.config.max_position_embeddings,
            "device": str(self._model.device),
            "dtype": str(self._model.dtype),
            "parameters": sum(p.numel() for p in self._model.parameters()),
        }

        if torch.cuda.is_available():
            info["gpu_memory"] = torch.cuda.memory_allocated()

        return info

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        streaming: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Gera texto com o modelo

        Args:
            prompt: Texto de entrada
            max_tokens: Número máximo de tokens a gerar
            temperature: Temperatura para sampling
            top_p: Probabilidade cumulativa para nucleus sampling
            top_k: Número de tokens para top-k sampling
            repetition_penalty: Penalidade para repetição
            do_sample: Se deve usar sampling
            streaming: Se deve retornar um gerador

        Returns:
            Texto gerado ou gerador de chunks
        """
        if self._model is None:
            raise RuntimeError("Modelo não carregado")

        try:
            # Tokenizar entrada
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

            # Configurar parâmetros de geração
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "pad_token_id": self._tokenizer.eos_token_id,
            }

            if streaming:

                def generate_stream():
                    for output in self._model.generate(
                        **inputs,
                        **gen_kwargs,
                        return_dict_in_generate=True,
                        output_scores=True,
                    ):
                        token = output.sequences[0, -1]
                        text = self._tokenizer.decode([token], skip_special_tokens=True)
                        yield text

                return generate_stream()

            else:
                outputs = self._model.generate(**inputs, **gen_kwargs)
                return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            self.logger.error(f"Erro na geração: {e}")
            raise

    def __enter__(self):
        """Context manager para carregamento automático"""
        self._load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager para descarregamento automático"""
        self._unload_model()
