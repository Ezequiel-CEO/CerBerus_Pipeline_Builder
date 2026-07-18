import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import gc
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PhiPro:
    """
    Implementação profissional e simples para o modelo Phi-2
    """

    def __init__(self, model_id: str = "microsoft/phi-2"):
        """
        Inicializa o loader do Phi-2.

        Args:
            model_id: ID do modelo no Hugging Face
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

        logger.info(f"PhiPro inicializado: {model_id} no dispositivo {self.device}")

    def load(self) -> None:
        """
        Carrega o modelo e tokenizer - simples e direto.
        """
        try:
            # Carregar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            logger.info(f"Tokenizer carregado: {self.model_id}")

            # Carregar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,  # Usar precisão FP16 padrão
                device_map="auto",  # Deixar o HF gerenciar os dispositivos automaticamente
                trust_remote_code=True,  # Necessário para modelos da Microsoft
            )

            # Informações pós-carregamento
            memory_usage = (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0
            )
            logger.info(f"Modelo carregado com sucesso: {self.model_id}")
            logger.info(f"Uso de memória GPU: {memory_usage:.2f}GB")

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise

    def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.7
    ) -> str:
        """
        Gera texto a partir de um prompt.

        Args:
            prompt: Texto de entrada
            max_tokens: Número máximo de tokens a gerar
            temperature: Temperatura para sampling (0.0-1.0)

        Returns:
            Texto gerado
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Modelo não carregado. Chame load() primeiro.")

        try:
            # Tokenizar entrada
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Gerar texto
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )

            # Decodificar resposta
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remover prompt da resposta
            if response.startswith(prompt):
                response = response[len(prompt) :].strip()

            # Limpar memória CUDA
            self._clear_cuda_memory()

            return response

        except Exception as e:
            logger.error(f"Erro ao gerar texto: {e}")
            raise

    def _clear_cuda_memory(self) -> None:
        """
        Limpa a memória CUDA.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def get_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo.

        Returns:
            Dicionário com informações do modelo
        """
        if self.model is None:
            return {"status": "não carregado"}

        memory_usage = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0
        )

        return {
            "model_id": self.model_id,
            "device": self.device,
            "precision": "fp16",
            "memory_usage": f"{memory_usage:.2f}GB",
        }

    def unload(self) -> None:
        """
        Descarrega o modelo da memória.
        """
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self._clear_cuda_memory()
        logger.info(f"Modelo {self.model_id} descarregado da memória")
