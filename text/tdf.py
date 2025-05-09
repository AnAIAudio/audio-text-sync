import os


def run_tdf(text_file_path: str):
    if not os.path.exists(text_file_path):
        raise FileNotFoundError(f"Text file not found: {text_file_path}")

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA

    # 예제 문장
    text_data = [
        "I love machine learning.",
        "Machine learning is fascinating.",
        "I love deep learning too.",
        "Learning never stops."
    ]

    with open(text_file_path, "r") as f:
        text_data = f.readlines()

    # 1. TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data).toarray()  # shape: (4, N)

    # 2. 차원 축소 (PCA)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(tfidf_matrix)  # shape: (4, 2)

    # 3. 시각화 (2D 시계열)
    plt.figure(figsize=(8, 6))
    plt.plot(reduced[:, 0], reduced[:, 1], marker='o', linestyle='-')

    # 문장 인덱스 표시
    for i, (x, y) in enumerate(reduced):
        plt.text(x + 0.01, y + 0.01, f"t{i + 1}")

    plt.title("Sentence Embedding Sequence (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()
