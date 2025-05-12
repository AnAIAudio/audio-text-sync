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


def map_time_code(sentences: list[str], alignment, srt_file_path: str):

    # 1. 문장 개수
    n_sentences = len(sentences)

    # 2. 문장별로 매핑된 오디오 프레임들 추출
    frame_to_time = lambda idx: idx * 0.02  # 20ms per frame (Wav2Vec2 default stride)

    sentence_times = []
    for i in range(n_sentences):
        indices = [j for j, k in zip(alignment.index1, alignment.index2) if j == i]
        if not indices:
            sentence_times.append((0.0, 0.0))
        else:
            audio_indices = [
                k for j, k in zip(alignment.index1, alignment.index2) if j == i
            ]
            start_time = frame_to_time(min(audio_indices))
            end_time = frame_to_time(max(audio_indices))
            sentence_times.append((start_time, end_time))

    # 2. 보정: 최소 길이 보장 & 중복 방지
    min_duration = 0.5  # 최소 자막 길이 0.5초
    gap = 0.1  # 문장 사이 최소 간격

    for i in range(len(sentence_times)):
        start, end = sentence_times[i]
        if end - start < min_duration:
            end = start + min_duration
        if i > 0:
            prev_end = sentence_times[i - 1][1]
            if start < prev_end + gap:
                start = prev_end + gap
                end = max(end, start + min_duration)
        sentence_times[i] = (round(start, 3), round(end, 3))

    # 3. SRT 파일 작성
    with open(srt_file_path, "w", encoding="utf-8") as f:
        for idx, (start, end) in enumerate(sentence_times):
            f.write(f"{idx+1}\n")
            f.write(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}\n")
            f.write(f"{sentences[idx]}\n\n")
    # 3. SRT 형식 출력
    print("\nGenerated SRT:")
    for idx, (start, end) in enumerate(sentence_times):
        print(f"{idx + 1}")
        print(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}")
        print(sentences[idx])
        print()


def run_to_srt(tokens, mapping, srt_file_path: str):
    import datetime

    # === Step 4. 각 토큰별 시간 추정 ===
    # 시간 변환 함수

    # 토큰별 오디오 프레임 매핑
    token_frames = {i: [] for i in range(len(tokens))}
    for i, j in mapping:
        token_frames[i].append(j)

    # SRT용 타임 코드 추정
    srt_entries = []
    for idx in range(len(tokens)):
        frames = token_frames[idx]
        if not frames:
            continue
        start_time = min(frames) * 0.02
        end_time = max(frames) * 0.02 + 0.3  # 끝은 약간 여유
        srt_entries.append(
            (
                idx + 1,
                seconds_to_srt_time(start_time),
                seconds_to_srt_time(end_time),
                tokens[idx],
            )
        )

    # === Step 5. SRT 저장 ===
    with open(srt_file_path, "w", encoding="utf-8") as f:
        for num, start, end, word in srt_entries:
            f.write(f"{num}\n{start} --> {end}\n{word}\n\n")
