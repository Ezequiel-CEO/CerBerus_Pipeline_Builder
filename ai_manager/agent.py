#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agente Inteligente para o CerBerusFMK
"""

import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
from pathlib import Path
import os

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuração avançada do agente"""

    model_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    max_context_length: int = 10
    use_8bit: bool = True
    use_4bit: bool = False
    offload_folder: Optional[str] = None
    trust_remote_code: bool = True


class Agent:
    """
    Agente inteligente baseado em LLM com gerenciamento avançado de recursos
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Inicializa o agente com configuração avançada"""
        self.config = config or AgentConfig()
        self._model = None
        self._tokenizer = None
        self._context = []
        self._model_path = Path(self.config.model_path)

        # Configurar logging detalhado
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Verificar GPU
        if torch.cuda.is_available():
            self.logger.info(f"GPU disponível: {torch.cuda.get_device_name(0)}")
            self.logger.info(
                f"Memória GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB"
            )
            self.logger.info(
                f"Memória GPU livre: {torch.cuda.memory_allocated() / 1024**3:.2f}GB"
            )

            # Limpar cache GPU
            torch.cuda.empty_cache()
        else:
            self.logger.warning("GPU não disponível, usando CPU")

        # Carregar modelo e tokenizer
        self._load_model()

    def _load_model(self) -> None:
        """Carrega o modelo e tokenizer com otimizações avançadas"""
        try:
            self.logger.info(f"Carregando modelo do agente: {self._model_path}")

            # Carregar tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._model_path), trust_remote_code=self.config.trust_remote_code
            )

            # Configurar carregamento otimizado do modelo
            model_kwargs = {
                "device_map": "auto",
                "trust_remote_code": self.config.trust_remote_code,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16,
            }

            # Configurar quantização
            if self.config.use_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
            elif self.config.use_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )

            # Carregar modelo
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self._model_path), **model_kwargs
            )

            self.logger.info("Modelo do agente carregado com sucesso")

        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo do agente: {e}")
            raise

    def _format_prompt(self, message: str) -> str:
        """Formata o prompt com contexto avançado"""
        # Adicionar mensagem ao contexto
        self._context.append(message)

        # Manter apenas as últimas mensagens
        if len(self._context) > self.config.max_context_length:
            self._context = self._context[-self.config.max_context_length :]

        # Formatar prompt com contexto estruturado
        prompt = """Você é um assistente profissional e útil chamado Cerberus. 
Sua função é ajudar usuários com suas dúvidas e tarefas.
Você deve responder sempre em português de forma clara e concisa.
Mantenha um tom profissional e amigável.

Histórico da conversa:
"""

        # Adicionar histórico de contexto
        for i, msg in enumerate(self._context):
            if i % 2 == 0:
                prompt += f"Usuário: {msg}\n"
            else:
                prompt += f"Assistente: {msg}\n"

        # Adicionar instrução para resposta
        if len(self._context) % 2 == 0:
            prompt += "Assistente:"

        return prompt

    def respond(self, message: str) -> str:
        """Gera uma resposta coerente e contextual"""
        if self._model is None:
            raise RuntimeError("Modelo não carregado")

        try:
            # Formatar prompt
            prompt = self._format_prompt(message)

            # Tokenizar entrada
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

            # Configurar geração avançada
            gen_config = GenerationConfig(
                max_new_tokens=150,  # Aumentado para respostas mais completas
                temperature=0.3,  # Reduzido para mais coerência
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                min_length=20,
                max_length=2048,
                num_beams=1,
                early_stopping=False,
            )

            # Gerar resposta
            outputs = self._model.generate(**inputs, generation_config=gen_config)

            # Decodificar resposta
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extrair apenas a parte nova da resposta
            response = response[len(prompt) :].strip()

            # Limpar resposta
            response = response.split("Usuário:")[0].strip()

            # Adicionar resposta ao contexto
            self._context.append(response)

            return response

        except Exception as e:
            self.logger.error(f"Erro ao gerar resposta: {e}")
            raise

    def clear_context(self) -> None:
        """Limpa o contexto do agente"""
        self._context = []

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
            "device": str(self._model.device),
            "dtype": str(self._model.dtype),
            "parameters": sum(p.numel() for p in self._model.parameters()),
        }

        # Adicionar atributos específicos do modelo se disponíveis
        for attr in ["max_position_embeddings", "n_positions", "max_sequence_length"]:
            if hasattr(self._model.config, attr):
                info["max_sequence_length"] = getattr(self._model.config, attr)
                break

        if torch.cuda.is_available():
            info["gpu_memory"] = torch.cuda.memory_allocated()

        return info

    def __enter__(self):
        """Context manager para carregamento automático"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager para limpeza"""
        self.clear_context()
        if self._model is not None:
            del self._model
            torch.cuda.empty_cache()

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Lista modelos disponíveis para uso.

        Returns:
            Lista de informações sobre os modelos
        """
        if self.llm_manager is None:
            self.logger.warning("Gerenciador de modelos não disponível")
            return []

        # Verificar diretório de modelos personalizado em ~/.cerberusfmk/models/
        custom_models_dir = os.path.expanduser("~/.cerberusfmk/models/")
        if os.path.exists(custom_models_dir):
            # Buscar modelos GGUF no diretório customizado
            self.logger.info(f"Verificando modelos em: {custom_models_dir}")
            for file in os.listdir(custom_models_dir):
                if file.endswith(".gguf"):
                    model_path = os.path.join(custom_models_dir, file)
                    self.llm_manager.add_model(model_path)

        # Obter lista de modelos como dicionários
        models_list = []
        try:
            models = self.llm_manager.list_models()

            # Se models for um dicionário, convertê-lo para lista
            if isinstance(models, dict):
                for model_id, model_info in models.items():
                    if isinstance(model_info, dict):
                        models_list.append(model_info)
            # Se já for uma lista, usar diretamente
            elif isinstance(models, list):
                models_list = models
        except Exception as e:
            self.logger.error(f"Erro ao listar modelos: {str(e)}")

        return models_list


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Configurar agente com modelo local
        config = AgentConfig(
            model_path="~/.cerberusfmk/models/qwen",  # Caminho para seu modelo Qwen
            max_length=2048,
            temperature=0.8,  # Temperatura reduzida para mais coerência
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            do_sample=True,
            max_context_length=5,
            use_4bit=True,  # Usar quantização 8-bit para economia de memória
        )

        # Inicializar agente
        with Agent(config) as agent:
            logger.info("Testando agente...")

            try:
                # Testar informações do modelo
                logger.info("\nInformações do modelo:")
                model_info = agent.get_model_info()
                for key, value in model_info.items():
                    logger.info(f"{key}: {value}")

                # Testar resposta em português
                test_message = "Olá! Como você pode me ajudar hoje?"
                logger.info(f"\nTestando resposta para: {test_message}")
                response = agent.respond(test_message)
                logger.info(f"Resposta: {response}")

                # Testar contexto
                test_message2 = "Você pode me explicar como funciona o sistema de versionamento de modelos?"
                logger.info(f"\nTestando segunda resposta para: {test_message2}")
                response2 = agent.respond(test_message2)
                logger.info(f"Resposta: {response2}")

                # Limpar contexto
                agent.clear_context()

            except Exception as e:
                logger.error(f"Erro durante o teste: {e}")
                raise

    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        raise
    finally:
        # Garantir limpeza de memória
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
