class AlignmentResult:
    def __init__(self, index1, index2, cost_matrix):
        # DTW 경로 좌표
        self.index1 = index1
        self.index2 = index2
        # 비용 행렬
        self.costMatrix = cost_matrix


def seconds_to_srt_time(seconds):
    import datetime

    td = datetime.timedelta(seconds=seconds)
    s = str(td)[:-3] if "." in str(td) else str(td) + ".000"
    if len(s.split(":")[0]) == 1:
        s = "0" + s
    return s.replace(".", ",")


def compare_dtw(text_embeds, audio_embeds):
    from dtw import dtw
    import numpy as np
    from scipy.spatial.distance import cosine

    # (예시) DTW 계산 초기화
    n_text = text_embeds.shape[0]
    n_audio = audio_embeds.shape[0]
    cost_matrix = np.zeros((n_text, n_audio), dtype=np.float32)

    # 1. 비용 계산 (간단히 L2 거리로 예시화)
    for i in range(n_text):
        for j in range(n_audio):
            cost_matrix[i, j] = np.linalg.norm(text_embeds[i] - audio_embeds[j])

    # 2. 누적 비용 dp_matrix 생성
    dp_matrix = np.zeros_like(cost_matrix)
    dp_matrix[0, 0] = cost_matrix[0, 0]

    for i in range(1, n_text):
        dp_matrix[i, 0] = cost_matrix[i, 0] + dp_matrix[i - 1, 0]
    for j in range(1, n_audio):
        dp_matrix[0, j] = cost_matrix[0, j] + dp_matrix[0, j - 1]

    for i in range(1, n_text):
        for j in range(1, n_audio):
            dp_matrix[i, j] = cost_matrix[i, j] + min(
                dp_matrix[i - 1, j],  # 위
                dp_matrix[i, j - 1],  # 왼
                dp_matrix[i - 1, j - 1],  # 대각
            )

    # 3. 경로 추적(backtrace)로 index1, index2 획득
    path_i = []
    path_j = []

    i = n_text - 1
    j = n_audio - 1
    path_i.append(i)
    path_j.append(j)

    while i > 0 and j > 0:
        neighbors = [
            dp_matrix[i - 1, j],
            dp_matrix[i, j - 1],
            dp_matrix[i - 1, j - 1],
        ]
        arg_min = np.argmin(neighbors)
        if arg_min == 0:
            i -= 1
        elif arg_min == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
        path_i.append(i)
        path_j.append(j)

    while i > 0:
        i -= 1
        path_i.append(i)
        path_j.append(j)

    while j > 0:
        j -= 1
        path_j.append(j)
        path_i.append(i)

    path_i.reverse()
    path_j.reverse()

    # AlignmentResult에 비용 행렬을 함께 저장
    return AlignmentResult(path_i, path_j, cost_matrix)

    # 3. 거리 계산 함수 정의 (cosine distance)
    def cosine_dist(x, y):
        return cosine(x, y)

    # 4. DTW 계산
    alignment = dtw(text_embeds, audio_embeds, dist_method=cosine_dist)

    return alignment


def run_dtw(text_embeds, audio_embeds):
    from dtw import dtw
    from scipy.spatial.distance import cosine

    # 3. 거리 계산 함수 정의 (cosine distance)
    def cosine_dist(x, y):
        return cosine(x, y)

    # 4. DTW 계산
    alignment = dtw(
        text_embeds,
        audio_embeds,
        dist_method=cosine_dist,
        keep_internals=True,
    )

    # 시각화용 변수
    # costMatrix = alignment.costMatrix  # (T_text+1, T_audio+1)  패딩 포함
    # index1 = alignment.index1  # 길이 = 정렬 경로 길이
    # index2 = alignment.index2
    # text_len = text_embeds.shape[0]  # T_text
    # audio_len = audio_embeds.shape[0]  # T_audio

    return alignment
