# -*- coding: utf-8 -*-
"""
Módulo para inferência nativa de modelos LLM
Implementa forward pass e geração de texto
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger("inference")


class KVCache:
    """
    Cache para armazenar valores de atenção key-value (KV) durante a geração
    Otimiza a inferência ao evitar recalcular estados para tokens já processados
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype=np.float32,
    ):
        """
        Inicializa o cache KV.

        Args:
            n_layers: Número de camadas do modelo
            n_heads: Número de cabeças de atenção
            head_dim: Dimensão de cada cabeça
            max_seq_len: Comprimento máximo da sequência
            dtype: Tipo de dados
        """
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        # Inicializar cache para K e V
        # Forma: [n_layers, 2, n_heads, max_seq_len, head_dim]
        # onde 2 é para K e V
        self.cache = np.zeros(
            (n_layers, 2, n_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.current_len = 0

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray, pos: int):
        """
        Atualiza o cache com novos valores K e V.

        Args:
            layer_idx: Índice da camada
            k: Tensor de chave (key)
            v: Tensor de valor (value)
            pos: Posição na sequência
        """
        if pos >= self.max_seq_len:
            raise ValueError(
                f"Posição {pos} excede o tamanho máximo do cache {self.max_seq_len}"
            )

        # Atualizar cache
        self.cache[layer_idx, 0, :, pos, :] = k
        self.cache[layer_idx, 1, :, pos, :] = v

    def get_kv(self, layer_idx: int, end_pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtém K e V do cache até a posição atual.

        Args:
            layer_idx: Índice da camada
            end_pos: Posição final (exclusiva)

        Returns:
            Tupla (K, V) com valores até a posição
        """
        k = self.cache[layer_idx, 0, :, :end_pos, :]
        v = self.cache[layer_idx, 1, :, :end_pos, :]
        return k, v

    def reset(self):
        """Limpa o cache"""
        self.cache.fill(0)
        self.current_len = 0


class Sampler:
    """
    Implementa estratégias de amostragem para geração de texto
    Suporta temperatura, top-k, top-p (nucleus), e repetition penalty
    """

    @staticmethod
    def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
        """
        Aplica temperatura aos logits.

        Args:
            logits: Array de logits
            temperature: Valor de temperatura (>0)

        Returns:
            Logits ajustados pela temperatura
        """
        if temperature <= 0:
            raise ValueError("A temperatura deve ser positiva")

        if temperature == 1.0:
            return logits

        return logits / temperature

    @staticmethod
    def apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
        """
        Aplica filtragem top-k aos logits.

        Args:
            logits: Array de logits
            k: Número de tokens de maior probabilidade a manter

        Returns:
            Logits com apenas os k maiores valores
        """
        if k <= 0:
            return logits

        # Encontrar valores e índices dos k maiores logits
        top_k_logits = np.sort(logits)[-k:]
        min_value = top_k_logits[0]

        # Criar máscara para zerar logits menores
        mask = logits >= min_value

        # Aplicar máscara
        filtered_logits = np.where(mask, logits, -float("inf"))

        return filtered_logits

    @staticmethod
    def apply_top_p(logits: np.ndarray, p: float) -> np.ndarray:
        """
        Aplica filtragem top-p (nucleus sampling) aos logits.

        Args:
            logits: Array de logits
            p: Probabilidade cumulativa limiar (0-1)

        Returns:
            Logits com apenas os tokens cuja probabilidade cumulativa é <= p
        """
        if p <= 0 or p > 1:
            raise ValueError("p deve estar no intervalo (0, 1]")

        if p == 1.0:
            return logits

        # Converter para probabilidades
        probs = Sampler.softmax(logits)

        # Ordenar índices por probabilidade decrescente
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Calcular probabilidade cumulativa
        cumulative_probs = np.cumsum(sorted_probs)

        # Criar máscara para tokens com probabilidade cumulativa <= p
        mask_indices = sorted_indices[cumulative_probs <= p]

        # Se nenhum token for selecionado, manter apenas o de maior probabilidade
        if len(mask_indices) == 0:
            mask_indices = [sorted_indices[0]]

        # Criar máscara binária
        mask = np.zeros_like(logits, dtype=bool)
        mask[mask_indices] = True

        # Aplicar máscara
        filtered_logits = np.where(mask, logits, -float("inf"))

        return filtered_logits

    @staticmethod
    def apply_repetition_penalty(
        logits: np.ndarray, tokens: List[int], penalty: float
    ) -> np.ndarray:
        """
        Aplica penalidade de repetição aos logits.

        Args:
            logits: Array de logits
            tokens: Lista de tokens já gerados
            penalty: Fator de penalidade (1.0 = sem penalidade, >1.0 = reduzir probabilidade)

        Returns:
            Logits com penalidade aplicada
        """
        if penalty == 1.0 or not tokens:
            return logits

        # Conjunto de tokens únicos já gerados
        unique_tokens = set(tokens)

        # Criar cópia dos logits para modificar
        penalized_logits = logits.copy()

        # Aplicar penalidade a cada token repetido
        for token in unique_tokens:
            if token < len(penalized_logits):
                if penalized_logits[token] > 0:
                    penalized_logits[token] /= penalty
                else:
                    penalized_logits[token] *= penalty

        return penalized_logits

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """
        Aplica função softmax para converter logits em probabilidades.

        Args:
            x: Array de logits

        Returns:
            Array de probabilidades
        """
        # Normalizar para evitar overflow/underflow
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def sample_token(logits: np.ndarray) -> int:
        """
        Amostra um token baseado nos logits.

        Args:
            logits: Array de logits

        Returns:
            Índice do token amostrado
        """
        # Converter para probabilidades
        probs = Sampler.softmax(logits)

        # Amostrar de acordo com as probabilidades
        return np.random.choice(len(probs), p=probs)

    @staticmethod
    def generate_next_token(
        logits: np.ndarray,
        prev_tokens: List[int],
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> int:
        """
        Gera o próximo token usando várias estratégias de amostragem.

        Args:
            logits: Array de logits para o próximo token
            prev_tokens: Tokens anteriores gerados
            temperature: Temperatura para amostragem
            top_k: Número de tokens com maior probabilidade a considerar (0 = desativado)
            top_p: Probabilidade cumulativa para nucleus sampling (1.0 = desativado)
            repetition_penalty: Penalidade para tokens repetidos (1.0 = sem penalidade)

        Returns:
            Índice do próximo token
        """
        # Aplicar temperatura
        logits = Sampler.apply_temperature(logits, temperature)

        # Aplicar penalidade de repetição
        logits = Sampler.apply_repetition_penalty(
            logits, prev_tokens, repetition_penalty
        )

        # Aplicar top-k
        if top_k > 0:
            logits = Sampler.apply_top_k(logits, top_k)

        # Aplicar top-p
        if top_p < 1.0:
            logits = Sampler.apply_top_p(logits, top_p)

        # Amostrar token
        return Sampler.sample_token(logits)


class NativeInference:
    """Implementação nativa de inferência para modelos LLM"""

    def __init__(self):
        self.model = None
        self.kv_cache = None

    def setup_model(self, weights: Dict[str, np.ndarray], config: Dict[str, Any]):
        """
        Configura o modelo para inferência.

        Args:
            weights: Dicionário de tensores do modelo
            config: Configuração do modelo
        """
        self.model = {"weights": weights, "config": config}

        # Configurar KV cache
        n_layers = config.get("block_count", 32)
        n_heads = config.get("head_count", 32)
        head_dim = config.get("embedding_length", 4096) // n_heads
        max_seq_len = config.get("context_length", 2048)

        self.kv_cache = KVCache(
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
        )

    def forward(self, input_ids: List[int]) -> np.ndarray:
        """
        Executa o forward pass do modelo para os tokens de entrada.

        Args:
            input_ids: Lista de IDs de token

        Returns:
            Logits para o próximo token
        """
        # Verificar se o modelo está configurado
        if not self.model:
            raise ValueError("Modelo não configurado. Chame setup_model primeiro.")

        # Implementação simulada para demonstração
        # Uma implementação real executaria o modelo completo

        # Simular camada de embedding
        embedding_weights = self.model["weights"].get("token_embd", None)
        if embedding_weights is not None:
            # Buscar embeddings para cada token de entrada
            embeddings = np.array(
                [
                    embedding_weights[token_id % len(embedding_weights)]
                    for token_id in input_ids
                ]
            )
        else:
            # Simular embeddings aleatórios se não temos os pesos
            embed_dim = self.model["config"].get("embedding_length", 4096)
            embeddings = np.random.randn(len(input_ids), embed_dim)

        # Simular logits para todos os tokens possíveis no vocabulário
        vocab_size = self.model["config"].get("vocab_size", 32000)
        logits = np.random.randn(vocab_size)

        # Simular tendência para próximos tokens relacionados
        if len(input_ids) > 0:
            last_token = input_ids[-1]
            # Aumentar probabilidade para tokens próximos ao último
            for i in range(max(0, last_token - 10), min(vocab_size, last_token + 10)):
                logits[i] += 2.0 * (1.0 - abs(i - last_token) / 10.0)

        return logits

    def generate(
        self,
        input_ids: List[int],
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        stop_ids: List[int] = None,
    ) -> List[int]:
        """
        Gera tokens de texto a partir dos tokens de entrada.

        Args:
            input_ids: Lista de IDs de token inicial
            max_length: Número máximo de tokens a gerar
            temperature: Temperatura para amostragem
            top_k: Número de tokens com maior probabilidade a considerar
            top_p: Probabilidade cumulativa para nucleus sampling
            repetition_penalty: Penalidade para tokens repetidos
            stop_ids: Lista de IDs de token que indicam o fim da geração

        Returns:
            Lista de IDs de token gerados
        """
        # Verificar se o modelo está configurado
        if not self.model:
            raise ValueError("Modelo não configurado. Chame setup_model primeiro.")

        # Inicializar com tokens de entrada
        generated_ids = input_ids.copy()

        # Inicializar cache KV
        if self.kv_cache:
            self.kv_cache.reset()

        # Lista de stop tokens
        if stop_ids is None:
            stop_ids = []

        # Adicionar EOS se presente na configuração
        eos_token_id = self.model["config"].get("eos_token_id", None)
        if eos_token_id is not None and eos_token_id not in stop_ids:
            stop_ids.append(eos_token_id)

        # Gerar tokens até max_length ou encontrar stop token
        for _ in range(max_length):
            # Executar modelo para obter logits
            logits = self.forward(generated_ids)

            # Gerar próximo token
            next_token = Sampler.generate_next_token(
                logits,
                generated_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            # Adicionar token gerado
            generated_ids.append(next_token)

            # Verificar se é um token de parada
            if next_token in stop_ids:
                break

        # Remover tokens de entrada e retornar apenas os gerados
        return generated_ids[len(input_ids) :]


def create_inference_engine(
    weights: Dict[str, np.ndarray], config: Dict[str, Any]
) -> NativeInference:
    """
    Cria um motor de inferência nativo.

    Args:
        weights: Dicionário de tensores do modelo
        config: Configuração do modelo

    Returns:
        Motor de inferência nativo
    """
    engine = NativeInference()
    engine.setup_model(weights, config)
    return engine
