"""
AletheionTokenizer: Wrapper unificado para tokenizacao.

Suporta:
- tiktoken (GPT-2, GPT-4, cl100k_base)
- sentencepiece (Llama-style)
- Vocabulario custom (treino de BPE proprio)

Para treinamento em escala, pre-tokeniza dados para formato binario
(numpy memmap) evitando re-tokenizacao a cada epoch.
"""

import os
import struct
import numpy as np
from typing import List, Optional, Union
from pathlib import Path


class AletheionTokenizer:
    """Tokenizer unificado para AletheionV2.

    Abstrai tiktoken/sentencepiece com interface unica.
    Suporta tokens especiais e pre-tokenizacao para binario.

    Args:
        name: Nome do encoding tiktoken ("gpt2", "cl100k_base", "o200k_base")
        custom_path: Caminho para modelo sentencepiece (.model)
    """

    # Tokens especiais
    PAD_TOKEN = "<|pad|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|endoftext|>"

    def __init__(
        self,
        name: str = "gpt2",
        custom_path: str = "",
    ):
        self.name = name
        self.custom_path = custom_path
        self._enc = None
        self._sp = None

        if custom_path and os.path.exists(custom_path):
            self._init_sentencepiece(custom_path)
        else:
            self._init_tiktoken(name)

    def _init_tiktoken(self, name: str) -> None:
        """Inicializa com tiktoken."""
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken nao instalado. Instale com: pip install tiktoken"
            )
        self._enc = tiktoken.get_encoding(name)
        self._backend = "tiktoken"

    def _init_sentencepiece(self, path: str) -> None:
        """Inicializa com sentencepiece."""
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "sentencepiece nao instalado. Instale com: pip install sentencepiece"
            )
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(path)
        self._backend = "sentencepiece"

    @property
    def vocab_size(self) -> int:
        """Tamanho do vocabulario."""
        if self._backend == "tiktoken":
            return self._enc.n_vocab
        return self._sp.GetPieceSize()

    @property
    def eos_id(self) -> int:
        """ID do token EOS."""
        if self._backend == "tiktoken":
            return self._enc.encode(
                self.EOS_TOKEN, allowed_special={self.EOS_TOKEN}
            )[0]
        return self._sp.eos_id()

    @property
    def bos_id(self) -> int:
        """ID do token BOS."""
        if self._backend == "tiktoken":
            # tiktoken GPT-2 nao tem BOS explicito
            return self.eos_id
        return self._sp.bos_id()

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Tokeniza texto para lista de IDs.

        Args:
            text: Texto para tokenizar
            add_bos: Adiciona BOS no inicio
            add_eos: Adiciona EOS no final

        Returns:
            Lista de token IDs
        """
        if self._backend == "tiktoken":
            ids = self._enc.encode(text, allowed_special={self.EOS_TOKEN})
        else:
            ids = self._sp.Encode(text)

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decodifica lista de IDs para texto.

        Args:
            ids: Lista de token IDs

        Returns:
            Texto decodificado
        """
        if self._backend == "tiktoken":
            return self._enc.decode(ids)
        return self._sp.Decode(ids)

    def encode_batch(
        self,
        texts: List[str],
        add_eos: bool = True,
    ) -> List[List[int]]:
        """Tokeniza batch de textos.

        Args:
            texts: Lista de textos
            add_eos: Adiciona EOS apos cada texto

        Returns:
            Lista de listas de IDs
        """
        results = []
        for text in texts:
            ids = self.encode(text, add_eos=add_eos)
            results.append(ids)
        return results

    def tokenize_to_file(
        self,
        texts,
        output_path: str,
        add_eos: bool = True,
        dtype: str = "uint16",
        verbose: bool = True,
    ) -> dict:
        """Tokeniza textos e salva como arquivo binario numpy.

        Formato: array 1D numpy memmap (uint16 para vocab < 65536, uint32 caso contrario).
        Isso permite leitura eficiente sem carregar tudo na RAM.

        Args:
            texts: Iteravel de strings (pode ser generator/streaming)
            output_path: Caminho do arquivo .bin de saida
            add_eos: Adiciona EOS entre documentos
            dtype: "uint16" (vocab <= 65535) ou "uint32"
            verbose: Mostra progresso

        Returns:
            Dict com estatisticas (total_tokens, total_docs, file_size_mb)
        """
        np_dtype = np.uint16 if dtype == "uint16" else np.uint32

        # Primeiro pass: tokeniza e acumula em chunks
        all_ids = []
        total_tokens = 0
        total_docs = 0
        chunk_size = 100_000  # Flush a cada 100k docs

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Arquivo temporario para acumular
        tmp_chunks = []

        for text in texts:
            if not text or not text.strip():
                continue

            ids = self.encode(text, add_eos=add_eos)
            all_ids.extend(ids)
            total_tokens += len(ids)
            total_docs += 1

            # Flush chunk para reduzir uso de RAM
            if len(all_ids) >= 10_000_000:  # 10M tokens
                chunk = np.array(all_ids, dtype=np_dtype)
                tmp_chunks.append(chunk)
                all_ids = []

                if verbose:
                    print(
                        f"  [TOKENIZE] {total_docs} docs, "
                        f"{total_tokens:,} tokens..."
                    )

        # Ultimo chunk
        if all_ids:
            tmp_chunks.append(np.array(all_ids, dtype=np_dtype))

        # Concatena e salva
        if not tmp_chunks:
            final = np.array([], dtype=np_dtype)
        else:
            final = np.concatenate(tmp_chunks)

        # Salva como arquivo binario raw
        final.tofile(str(output_path))

        # Salva metadata
        meta_path = str(output_path) + ".meta"
        meta = {
            "total_tokens": total_tokens,
            "total_docs": total_docs,
            "dtype": dtype,
            "vocab_size": self.vocab_size,
            "tokenizer": self.name,
        }
        import json
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        if verbose:
            print(
                f"  [TOKENIZE] Completo: {total_tokens:,} tokens, "
                f"{total_docs:,} docs, {file_size_mb:.1f} MB"
            )

        return {
            "total_tokens": total_tokens,
            "total_docs": total_docs,
            "file_size_mb": file_size_mb,
        }

    @staticmethod
    def load_token_file(
        path: str,
        dtype: str = "uint16",
    ) -> np.ndarray:
        """Carrega arquivo de tokens pre-tokenizados como memmap.

        Usa memmap para nao carregar tudo na RAM.

        Args:
            path: Caminho do arquivo .bin
            dtype: Tipo do array

        Returns:
            numpy memmap array 1D
        """
        np_dtype = np.uint16 if dtype == "uint16" else np.uint32
        return np.memmap(path, dtype=np_dtype, mode="r")

    def __repr__(self) -> str:
        return (
            f"AletheionTokenizer(backend={self._backend}, "
            f"name={self.name}, vocab_size={self.vocab_size})"
        )
