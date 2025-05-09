import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def dtw(x, y, band_width):
    n = len(x)
    m = len(y)

    dtw_matrix = np.inf * np.ones((n, m))
    dtw_matrix[0, 0] = abs(x[0] - y[0])  # 첫 번째 원소의 차이

    # 동적 프로그래밍 계산
    for i in range(1, n):
        for j in range(max(1, i - band_width), min(m, i + band_width + 1)):
            cost = abs(x[i] - y[j])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
            )

    # 최적 경로 추적
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_cost = min(
                dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
            )
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


# 예시 시계열
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 2, 3, 4, 5])

# 대역 크기 설정
band_width = 1

# DTW 계산
dtw_matrix, path = dtw(x, y, band_width)

# 시각화
plt.figure(figsize=(10, 6))

# DTW 비용 행렬 시각화
sns.heatmap(dtw_matrix, cmap="YlGnBu", cbar=True)
plt.title("DTW Distance Matrix with Sakoe-Chiba Band")
plt.xlabel("y (series 2)")
plt.ylabel("x (series 1)")

# 최적 경로 시각화
for i, j in path:
    plt.plot(j, i, marker="o", color="red", markersize=3)

plt.show()
