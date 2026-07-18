import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, Dict, Any
import logging
from pathlib import Path
import json
import os
import gc
import psutil

from .qwen_config import QwenConfig

logger = logging.getLogger(__name__)


class QwenLoader:
    """Loader específico para Qwen2.5-7B-Instruct"""

    def __init__(self, config: Optional[QwenConfig] = None):
        self.config = config or QwenConfig()
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Criar diretório para offload
        os.makedirs(self.config.offload_folder, exist_ok=True)

        # Configurar ambiente para otimizar memória
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Usar apenas a primeira GPU

    def _clean_memory(self):
        """Limpa a memória de forma agressiva"""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # Limpar cache CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Forçar coleta de lixo
        gc.collect()

        # Processo Python atual
        process = psutil.Process()
        logger.info(
            f"Uso de memória RAM: {process.memory_info().rss / (1024 * 1024):.2f} MB"
        )

    def load(self) -> None:
        """Carrega o modelo e tokenizer com configurações específicas"""
        try:
            # Limpar memória CUDA e RAM
            self._clean_memory()

            logger.info(f"Carregando Qwen2.5-7B-Instruct de {self.config.model_path}")

            # Carregar tokenizer primeiro, que é pequeno
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                use_fast=True,
                local_files_only=True,
                padding_side="left",
            )

            # Configurar quantização 8-bit
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=None,
                llm_int8_enable_fp32_cpu_offload=True,
            )

            # Configurar device map para distribuir o modelo entre GPU e CPU
            device_map = {
                "transformer.wte": "cpu",
                "transformer.ln_f": "cpu",
                "lm_head": "cpu",
            }

            # Distribuir as camadas do transformador
            num_layers = 28  # Número de camadas do modelo Qwen 7B
            gpu_layers = min(4, num_layers)  # Usar 4 camadas na GPU

            for i in range(num_layers):
                if i < gpu_layers:
                    device_map[f"transformer.h.{i}"] = 0  # GPU
                else:
                    device_map[f"transformer.h.{i}"] = "cpu"  # CPU

            # Configurar carregamento do modelo
            model_kwargs = {
                "device_map": device_map,
                "trust_remote_code": self.config.trust_remote_code,
                "torch_dtype": torch.float16,
                "use_cache": True,
                "use_safetensors": True,
                "low_cpu_mem_usage": True,
                "offload_folder": (
                    self.config.offload_folder if self.config.cpu_offload else None
                ),
                "max_memory": {0: "2GB", "cpu": "32GB"},
                "local_files_only": True,
                "quantization_config": quantization_config,
            }

            # Carregar modelo
            logger.info("Iniciando carregamento com offloading CPU...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path, **model_kwargs
            )

            # Relatório final
            logger.info("Modelo Qwen2.5-7B-Instruct carregado com sucesso")
            logger.info(f"Configuração: {self.config}")
            logger.info(f"Device Map: {device_map}")

            # Verificar uso de memória final
            if torch.cuda.is_available():
                logger.info(
                    f"Memória GPU usada após carregamento: {torch.cuda.memory_allocated() / 1024**3:.2f}GB"
                )

            # Processo Python atual
            process = psutil.Process()
            logger.info(
                f"Uso de memória RAM após carregamento: {process.memory_info().rss / (1024 * 1024):.2f} MB"
            )

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise

    def generate(self, prompt: str, **kwargs) -> str:
        """Gera texto com configurações específicas"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Modelo não carregado")

        try:
            # Tokenizar entrada
            inputs = self._tokenizer(prompt, return_tensors="pt")

            # Mover inputs para o dispositivo correto conforme o mapeamento
            if isinstance(self._model.device_map, dict):
                # Se estiver usando device_map personalizado, alguns inputs podem precisar ir para CPU
                # e o modelo vai lidar com a transferência interna
                pass
            else:
                # Se for um dispositivo único, mover para ele
                inputs = inputs.to(next(self._model.parameters()).device)

            # Configurar geração
            gen_kwargs = {
                "max_new_tokens": min(
                    self.config.max_length, 512
                ),  # Limitar a 512 para economizar memória
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repetition_penalty": self.config.repetition_penalty,
                "do_sample": self.config.do_sample,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
                **kwargs,
            }

            # Gerar resposta
            with torch.no_grad():  # Economia de memória durante a geração
                outputs = self._model.generate(**inputs, **gen_kwargs)

            # Limpar cache após geração
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Decodificar resposta
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remover o prompt original da resposta
            prompt_text = self._tokenizer.decode(
                inputs.input_ids[0], skip_special_tokens=True
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

        # Processo Python atual
        process = psutil.Process()
        ram_usage = process.memory_info().rss / (1024 * 1024 * 1024)  # GB

        return {
            "model_name": "Qwen2.5-7B-Instruct",
            "device": self._device,
            "quantization": (
                "8-bit"
                if self.config.use_8bit
                else "4-bit" if self.config.use_4bit else "none"
            ),
            "max_length": self.config.max_length,
            "temperature": self.config.temperature,
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
