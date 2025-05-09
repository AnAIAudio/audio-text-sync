def run_bert():
    # 설치 필요 (최초 1회만)
    # pip install sentence-transformers

    from sentence_transformers import SentenceTransformer
    import numpy as np

    # 1. BERT 모델 불러오기 (빠르고 가벼운 모델 사용)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. 문장 리스트 준비 (시계열 순서대로 정렬된 문장들)
    sentences = [
        "나는 커피를 좋아해.",
        "아침마다 커피를 마신다.",
        "오늘은 날씨가 흐리다.",
        "카페에 앉아 책을 읽고 있다."
    ]

    # 3. 문장 → BERT 임베딩 (각 문장은 하나의 시점)
    sentence_embeddings = model.encode(sentences)

    # 4. NumPy 배열로 변환 (DTW에 바로 사용 가능)
    time_series = np.array(sentence_embeddings)

    print(time_series.shape)  # 예: (4, 384) → 문장 4개, 각 벡터 384차원
    print(time_series[0])  # 첫 문장의 벡터