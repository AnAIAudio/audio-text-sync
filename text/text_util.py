def change_text_1st_dimension(text_file_path: str):
    import os
    if not os.path.exists(text_file_path):
        raise FileNotFoundError(f"Text file not found: {text_file_path}")

    import numpy as np

    text = "I love machine learning"
    with open(text_file_path, "r") as f:
        text = f.read()

    # 고유 문자 목록 생성
    chars = sorted(list(set(text)))  # 고유 문자 집합
    char_to_index = {char: idx for idx, char in enumerate(chars)}  # 문자 -> 인덱스 맵핑

    # 텍스트를 숫자 인덱스 리스트로 변환
    text_indices = [char_to_index[char] for char in text]

    # 1차원 numpy 배열로 변환
    text_array = np.array(text_indices)

    print(f"1차원 배열: {text_array}")
    return text_array
