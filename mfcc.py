import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def run_mfcc():

    # 오디오 파일 로드
    file_path = "path_to_audio_file.wav"  # 오디오 파일 경로
    y, sr = librosa.load(file_path, sr=None)  # sr=None으로 원본 샘플링 레이트 사용

    # MFCC 추출
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 13개의 MFCC 계수 추출

    # MFCC 시각화
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfccs, x_axis="time", sr=sr)
    plt.colorbar(format="%+2.0f dB")
    plt.title("MFCC (Mel Frequency Cepstral Coefficients)")
    plt.tight_layout()
    plt.show()
