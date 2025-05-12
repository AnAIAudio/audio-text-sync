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
