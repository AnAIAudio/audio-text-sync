import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from audio.audio_util import change_audio_1st_dimension
from text.text_util import text_to_array


# 사코에-치바 대역 DTW 구현 (비용 행렬과 최적 경로 추적)
def sakoe_chiba_dtw(x, y, band_width):
    n = len(x)
    m = len(y)

    # 비용 행렬 초기화: 0으로 설정하지 않도록 최소값을 추가 (예: 1e-6)
    dtw_matrix = np.inf * np.ones((n, m))
    dtw_matrix[0, 0] = abs(x[0] - y[0]) + 1e-6  # 작은 값 추가

    # 동적 프로그래밍 계산 (사코에-치바 대역 적용)
    for i in range(1, n):
        for j in range(max(1, i - band_width), min(m, i + band_width + 1)):
            cost = abs(x[i] - y[j]) + 1e-6  # 작은 값 추가
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

    # 최적 경로 추적
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_cost = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
            if min_cost == dtw_matrix[i - 1, j]:
                i -= 1
            elif min_cost == dtw_matrix[i, j - 1]:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))

    path.reverse()
    return dtw_matrix, path


def run_dtw(audio_file_path, text_file_path: str):
    x = change_audio_1st_dimension(audio_file_path=audio_file_path)
    y = text_to_array(text_file_path=text_file_path)
    # x = np.array([1, 2, 3, 4, 5])
    # y = np.array([2, 2, 3, 4, 5])

    # 큰 배열을 방지하기 위한 다운 샘플링 예시
    max_len = 10000  # 필요에 따라 조절
    if len(x) > max_len:
        step_x = max(len(x) // max_len, 1)
        x = x[::step_x]
    if len(y) > max_len:
        step_y = max(len(y) // max_len, 1)
        y = y[::step_y]

    band_width = 1  # 대역 크기 설정

    # 두 시계열 간의 DTW 거리 계산 및 비용 행렬과 경로 얻기
    dtw_matrix, path = sakoe_chiba_dtw(y, x, band_width)

    # 6. DTW 비용 행렬 시각화 (히트맵)
    plt.figure(figsize=(10, 6))
    sns.heatmap(dtw_matrix, cmap='YlGnBu', cbar=True)
    plt.title("DTW Distance Matrix with Sakoe-Chiba Band")
    plt.xlabel("Text Data (y)")
    plt.ylabel("Audio Data (x)")

    # 7. 최적 경로 시각화
    for (i, j) in path:
        plt.plot(j, i, marker="o", color="red", markersize=3)

    plt.show()
