def run_sentence_bert(text_list: list[str]):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # 1. BERT 모델 불러오기 (빠르고 가벼운 모델 사용)
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # model = SentenceTransformer("all-mpnet-base-v2")
    model = SentenceTransformer("bert-base-nli-mean-tokens")

    # 3. 문장 → BERT 임베딩 (각 문장은 하나의 시점)
    sentence_embeddings = model.encode(text_list)

    print("Text embeddings shape : ", sentence_embeddings.shape)

    return sentence_embeddings



from functools import lru_cache
from typing import List, Literal
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import torch

@lru_cache(maxsize=None)
def _load_model(model_name: str, device: str):
    return SentenceTransformer(model_name, device=device)

def run_token_level_bert(
        texts: List[str],
        model_name: str = "all-mpnet-base-v2",
        device: Literal["cpu", "cuda"] = "cuda",
        batch_size: int = 32,
        l2_norm: bool = True,
        ) -> np.ndarray:
    """
    texts  : 문장 리스트
    return : (Σ token 수, hidden_dim)  — 토큰 단위 시계열
    """
    model = _load_model(model_name, device)
    # token_embeddings = True → WordPiece 벡터 반환
    token_seqs = model.encode(
        texts,
        batch_size=batch_size,
        output_value="token_embeddings",
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 32,
    )                    # List[np.ndarray]; 각 배열 shape=(tok_in_sentence, D)

    # 하나의 긴 시계열로 연결
    embeds = np.concatenate(token_seqs, axis=0)   # (total_tokens, D)

    if l2_norm:
        embeds = normalize(embeds)

    print("Token-level embeddings shape :", embeds.shape)
    return embeds