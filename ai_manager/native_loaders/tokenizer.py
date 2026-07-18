# -*- coding: utf-8 -*-
"""
Tokenizador nativo para modelos LLM
Suporta tokenizações comuns: BPE, WordPiece, SentencePiece
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger("tokenizer")


class TokenizerBase:
    """Classe base para tokenizadores"""

    def __init__(self):
        self.vocab = {}
        self.ids_to_tokens = {}
        self.bos_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None
        self.unk_token_id = None

    def tokenize(self, text: str) -> List[int]:
        """Tokeniza o texto para IDs de token"""
        raise NotImplementedError("Método não implementado na classe base")

    def decode(self, token_ids: List[int]) -> str:
        """Converte IDs de token para texto"""
        raise NotImplementedError("Método não implementado na classe base")

    def load_vocab(self, vocab_path: str):
        """Carrega vocabulário de um arquivo"""
        raise NotImplementedError("Método não implementado na classe base")


class BPETokenizer(TokenizerBase):
    """Tokenizador BPE (Byte Pair Encoding)"""

    def __init__(
        self, vocab_path: Optional[str] = None, merges_path: Optional[str] = None
    ):
        super().__init__()
        self.merges = {}
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        if vocab_path:
            self.load_vocab(vocab_path)

        if merges_path:
            self.load_merges(merges_path)

    def _bytes_to_unicode(self) -> Dict[int, str]:
        """
        Mapeia bytes para caracteres Unicode para manipulação de texto
        Baseado na implementação OpenAI GPT
        """
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs.copy()
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}

    def load_vocab(self, vocab_path: str):
        """Carrega vocabulário de um arquivo JSON"""
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)

            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

            # Tentar detectar tokens especiais
            for token, id in self.vocab.items():
                if token in ["<s>", "<bos>"]:
                    self.bos_token_id = id
                elif token in ["</s>", "<eos>"]:
                    self.eos_token_id = id
                elif token in ["<pad>"]:
                    self.pad_token_id = id
                elif token in ["<unk>"]:
                    self.unk_token_id = id

            logger.info(f"Carregado vocabulário com {len(self.vocab)} tokens")
        except Exception as e:
            logger.error(f"Erro ao carregar vocabulário: {e}")

    def load_merges(self, merges_path: str):
        """Carrega pares de mesclagem BPE"""
        try:
            with open(merges_path, "r", encoding="utf-8") as f:
                merges = f.read().split("\n")

            merges = merges[1:] if merges[0].startswith("#") else merges
            merges = [tuple(merge.split()) for merge in merges if merge]
            self.merges = dict(zip(merges, range(len(merges))))

            logger.info(f"Carregadas {len(self.merges)} regras de mesclagem BPE")
        except Exception as e:
            logger.error(f"Erro ao carregar mesclagens BPE: {e}")

    def bpe(self, token: str) -> str:
        """Aplica o algoritmo BPE ao token"""
        if not self.merges:
            return token

        word = tuple(token)
        pairs = self._get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.merges.get(pair, float("inf")))
            if bigram not in self.merges:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i : i + 2] == bigram:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break

            pairs = self._get_pairs(word)

        return "".join(word)

    def _get_pairs(self, word: Tuple[str, ...]) -> set:
        """Retorna pares de símbolos adjacentes no token"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def tokenize(self, text: str) -> List[int]:
        """Tokeniza o texto para IDs de token"""
        tokens = []

        # Adicionar BOS se definido
        if self.bos_token_id is not None:
            tokens.append(self.bos_token_id)

        # Pré-processamento do texto
        text = text.lower()

        # Encode cada token
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_token = self.bpe(token)

            # Lookup no vocabulário
            if bpe_token in self.vocab:
                tokens.append(self.vocab[bpe_token])
            else:
                # Fallback para UNK ou caracteres individuais
                if self.unk_token_id is not None:
                    tokens.append(self.unk_token_id)
                else:
                    # Tokenização por caractere como fallback
                    for c in token:
                        if c in self.vocab:
                            tokens.append(self.vocab[c])
                        else:
                            # Último recurso: usar o código ASCII
                            tokens.append(ord(c) % 256)

        # Adicionar EOS se definido
        if self.eos_token_id is not None:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Converte IDs de token para texto"""
        text = []

        for token_id in token_ids:
            # Pular tokens especiais
            if token_id in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                continue

            if token_id in self.ids_to_tokens:
                text.append(self.ids_to_tokens[token_id])
            else:
                # Fallback para caractere ASCII
                text.append(chr(token_id % 256))

        # Decodificar bytes para UTF-8
        result = "".join(text)

        # Substituir os caracteres unicode pelos bytes correspondentes
        result = "".join([self.byte_decoder.get(c, c) for c in result])

        return result


class SimpleTokenizer(TokenizerBase):
    """Tokenizador simples baseado em caracteres ou palavras"""

    def __init__(self, vocab_path: Optional[str] = None, by_word: bool = False):
        super().__init__()
        self.by_word = by_word

        if vocab_path:
            self.load_vocab(vocab_path)

    def load_vocab(self, vocab_path: str):
        """Carrega vocabulário de um arquivo de texto"""
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = [line.strip() for line in f if line.strip()]

            self.vocab = {token: i for i, token in enumerate(vocab)}
            self.ids_to_tokens = {i: token for i, token in enumerate(vocab)}

            # Detectar tokens especiais
            for token, id in self.vocab.items():
                if token in ["<s>", "<bos>"]:
                    self.bos_token_id = id
                elif token in ["</s>", "<eos>"]:
                    self.eos_token_id = id
                elif token in ["<pad>"]:
                    self.pad_token_id = id
                elif token in ["<unk>"]:
                    self.unk_token_id = id

            logger.info(f"Carregado vocabulário simples com {len(self.vocab)} tokens")
        except Exception as e:
            logger.error(f"Erro ao carregar vocabulário: {e}")

    def tokenize(self, text: str) -> List[int]:
        """Tokeniza o texto por palavras ou caracteres"""
        tokens = []

        # Adicionar BOS se definido
        if self.bos_token_id is not None:
            tokens.append(self.bos_token_id)

        if self.by_word:
            # Tokenização por palavras
            words = text.split()
            for word in words:
                if word in self.vocab:
                    tokens.append(self.vocab[word])
                elif self.unk_token_id is not None:
                    tokens.append(self.unk_token_id)
                else:
                    # Fallback para tokenização por caractere
                    for c in word:
                        if c in self.vocab:
                            tokens.append(self.vocab[c])
                        else:
                            tokens.append(ord(c) % 256)
        else:
            # Tokenização por caracteres
            for c in text:
                if c in self.vocab:
                    tokens.append(self.vocab[c])
                elif self.unk_token_id is not None:
                    tokens.append(self.unk_token_id)
                else:
                    tokens.append(ord(c) % 256)

        # Adicionar EOS se definido
        if self.eos_token_id is not None:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Converte IDs de token para texto"""
        text = []

        for token_id in token_ids:
            # Pular tokens especiais
            if token_id in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                continue

            if token_id in self.ids_to_tokens:
                text.append(self.ids_to_tokens[token_id])
            else:
                # Fallback para caractere ASCII
                text.append(chr(token_id % 256))

        if self.by_word:
            return " ".join(text)
        else:
            return "".join(text)


def create_tokenizer(
    tokenizer_type: str,
    vocab_path: Optional[str] = None,
    merges_path: Optional[str] = None,
) -> TokenizerBase:
    """
    Cria um tokenizador de acordo com o tipo especificado.

    Args:
        tokenizer_type: Tipo de tokenizador (bpe, simple)
        vocab_path: Caminho para o arquivo de vocabulário
        merges_path: Caminho para o arquivo de mesclagens (apenas para BPE)

    Returns:
        Instância do tokenizador
    """
    if tokenizer_type.lower() == "bpe":
        return BPETokenizer(vocab_path, merges_path)
    else:
        return SimpleTokenizer(vocab_path, tokenizer_type.lower() == "word")
