import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, Dict, Any
import logging
from pathlib import Path
import json
import os
import gc
import psutil

logger = logging.getLogger(__name__)


class PhiLoader:
    """Loader otimizado para phi-2"""

    def __init__(self, model_path: str = "microsoft/phi-2"):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Configurar ambiente para otimizar memória
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def _clean_memory(self):
        """Limpa a memória de forma agressiva"""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        gc.collect()

        process = psutil.Process()
        logger.info(
            f"Uso de memória RAM: {process.memory_info().rss / (1024 * 1024):.2f} MB"
        )

    def load(self) -> None:
        """Carrega o modelo e tokenizer com configurações otimizadas"""
        try:
            self._clean_memory()

            logger.info(f"Carregando phi-2 de {self.model_path}")

            # Carregar tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=True,
                padding_side="left",
            )

            # Configurar quantização 8-bit
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=None,
                llm_int8_enable_fp32_cpu_offload=True,
            )

            # Configurar carregamento
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "use_cache": True,
                "use_safetensors": True,
                "low_cpu_mem_usage": True,
                "quantization_config": quantization_config,
                "device_map": "auto",  # Deixar o HuggingFace gerenciar
            }

            # Carregar modelo
            logger.info("Iniciando carregamento...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **model_kwargs
            )

            logger.info("Modelo phi-2 carregado com sucesso")

            if torch.cuda.is_available():
                logger.info(
                    f"Memória GPU usada: {torch.cuda.memory_allocated() / 1024**3:.2f}GB"
                )

            process = psutil.Process()
            logger.info(
                f"Uso de memória RAM: {process.memory_info().rss / (1024 * 1024):.2f} MB"
            )

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise

    def generate(self, prompt: str, **kwargs) -> str:
        """Gera texto com configurações otimizadas"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Modelo não carregado")

        try:
            # Tokenizar entrada e mover para GPU
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Configurar geração
            gen_kwargs = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
                **kwargs,
            }

            # Gerar resposta
            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)

            if torch.cuda.is_available():
                outputs = outputs.cpu()
                torch.cuda.empty_cache()

            # Decodificar resposta
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remover prompt
            prompt_text = self._tokenizer.decode(
                inputs["input_ids"][0], skip_special_tokens=True
            )
            if response.startswith(prompt_text):
                response = response[len(prompt_text) :].strip()

            return response

        except Exception as e:
            logger.error(f"Erro ao gerar texto: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo"""
        if self._model is None:
            return {}

        process = psutil.Process()
        ram_usage = process.memory_info().rss / (1024 * 1024 * 1024)

        return {
            "model_name": "phi-2",
            "device": self._device,
            "quantization": "8-bit",
            "gpu_memory": (
                f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"
                if torch.cuda.is_available()
                else "N/A"
            ),
            "ram_usage": f"{ram_usage:.2f}GB",
            "device_map": (
                str(self._model.device_map)
                if hasattr(self._model, "device_map")
                else "N/A"
            ),
        }
