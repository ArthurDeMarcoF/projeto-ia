"""Tokenização, vocabulário e conversão entre tokens e Token IDs."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

UNK_TOKEN = "<|unk|>"
EOT_TOKEN = "<|endoftext|>"


def tokenize_text(text: str) -> List[str]:
    """Tokenizador didático baseado na abordagem apresentada no Capítulo 2.

    Separa palavras, pontuação e o marcador ``--``. Espaços são usados apenas
    como separadores e não são mantidos como tokens.
    """
    pieces = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [piece.strip() for piece in pieces if piece.strip()]


def build_vocabulary(tokens: Iterable[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Cria os mapas token -> id e id -> token.

    Os tokens especiais são acrescentados ao final para manter o vocabulário
    explícito e permitir o tratamento de palavras desconhecidas e fim de texto.
    """
    unique_tokens = sorted(set(tokens))
    for special_token in (EOT_TOKEN, UNK_TOKEN):
        if special_token not in unique_tokens:
            unique_tokens.append(special_token)

    str_to_int = {token: idx for idx, token in enumerate(unique_tokens)}
    int_to_str = {idx: token for token, idx in str_to_int.items()}
    return str_to_int, int_to_str


class SimpleTokenizerV2:
    """Tokenizador didático com encode/decode e suporte a token desconhecido."""

    def __init__(self, str_to_int: Dict[str, int]):
        self.str_to_int = dict(str_to_int)
        self.int_to_str = {idx: token for token, idx in self.str_to_int.items()}

        if UNK_TOKEN not in self.str_to_int or EOT_TOKEN not in self.str_to_int:
            raise ValueError(
                f"O vocabulário precisa conter {UNK_TOKEN} e {EOT_TOKEN}."
            )

    @property
    def vocab_size(self) -> int:
        return len(self.str_to_int)

    def encode(self, text: str) -> List[int]:
        tokens = tokenize_text(text)
        unk_id = self.str_to_int[UNK_TOKEN]
        return [self.str_to_int.get(token, unk_id) for token in tokens]

    def decode(self, ids: Iterable[int]) -> str:
        tokens = [self.int_to_str[int(idx)] for idx in ids]
        text = " ".join(tokens)
        # Remove espaços artificiais antes de sinais de pontuação.
        return re.sub(r'\s+([,.?!”!";:()\'])', r'\1', text)


def create_simple_tokenizer(text: str) -> SimpleTokenizerV2:
    """Cria um tokenizador e seu vocabulário a partir de um corpus."""
    tokens = tokenize_text(text)
    str_to_int, _ = build_vocabulary(tokens)
    return SimpleTokenizerV2(str_to_int)


def get_gpt2_tokenizer():
    """Retorna o tokenizador BPE do GPT-2 via tiktoken.

    O import é local para que os demais componentes didáticos continuem
    executáveis mesmo antes da instalação opcional da biblioteca.
    """
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken não está instalado. Execute: pip install -r requirements.txt"
        ) from exc

    return tiktoken.get_encoding("gpt2")
