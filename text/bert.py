def run_bert(text_data: list[str]):
    # 설치 필요 (최초 1회만)
    # pip install sentence-transformers

    from sentence_transformers import SentenceTransformer
    import numpy as np

    # 1. BERT 모델 불러오기 (빠르고 가벼운 모델 사용)
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer("all-mpnet-base-v2")
    # all-mpnet-base-v2

    # 3. 문장 → BERT 임베딩 (각 문장은 하나의 시점)
    sentence_embeddings = model.encode(text_data)

    return sentence_embeddings

    # 4. NumPy 배열로 변환 (DTW에 바로 사용 가능)
    # time_series = np.array(sentence_embeddings)
