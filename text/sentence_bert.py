def run_sentence_bert(text_list: list[str]):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # 1. BERT 모델 불러오기 (빠르고 가벼운 모델 사용)
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer("all-mpnet-base-v2")
    # all-mpnet-base-v2

    # 3. 문장 → BERT 임베딩 (각 문장은 하나의 시점)
    sentence_embeddings = model.encode(text_list)

    # 4. NumPy 배열로 변환 (DTW에 바로 사용 가능)
    time_series = np.array(sentence_embeddings)

    print(time_series.shape)  # 예: (4, 384) → 문장 4개, 각 벡터 384차원
    print(time_series[0])  # 첫 문장의 벡터

    return time_series