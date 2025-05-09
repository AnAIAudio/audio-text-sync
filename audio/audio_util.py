def change_audio_1st_dimension(audio_file_path: str):
    import os
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    import librosa
    import numpy as np

    # librosa를 사용해 음원 파일을 읽고 1차원 배열로 변환
    y, sr = librosa.load(audio_file_path, sr=None)  # sr=None으로 지정하면 원본 샘플링 주파수 유지

    # 1차원 numpy 배열로 음원 데이터
    print(f"Sample Rate: {sr}")
    print(f"Data shape: {y.shape}")

    return y
