import importlib.util
import os
from typing import Optional, Union, List

import numpy as np
from sentence_transformers import SentenceTransformer


def _resolve_attn_implementation() -> str:
    requested = os.getenv("WORLDMM_TEXT_ATTN_IMPLEMENTATION") or os.getenv("WORLDMM_ATTN_IMPLEMENTATION")
    if requested:
        return requested
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def _resolve_device_map(device: str) -> str:
    return os.getenv("WORLDMM_TEXT_DEVICE_MAP", device)


def _resolve_text_batch_size(default: int = 8) -> int:
    raw_value = os.getenv("WORLDMM_TEXT_EMBED_BATCH_SIZE") or os.getenv("WORLDMM_EMBEDDING_BATCH_SIZE")
    if not raw_value:
        return default
    try:
        return max(1, int(raw_value))
    except ValueError:
        return default


class Qwen3EmbeddingModel:
    """Wrapper for Qwen3 Embedding Model"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={
                "attn_implementation": _resolve_attn_implementation(),
                "dtype": "auto",
                "device_map": _resolve_device_map(device),
            },
            tokenizer_kwargs={"padding_side": "left"},
        )
    
    def encode_text(self, texts: Union[str, List[str]], batch_size: Optional[int] = None) -> np.ndarray:
        """Encode text into embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = batch_size or _resolve_text_batch_size()
        embeddings = self.model.encode(texts, batch_size=batch_size)
        return embeddings
    
    def encode(self, content: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Universal encode method for text"""
        return self.encode_text(content, **kwargs)
