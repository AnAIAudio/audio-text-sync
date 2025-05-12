def compare_dtw(text_embeds, audio_embeds):
    from dtw import dtw
    from scipy.spatial.distance import cosine

    # 3. 거리 계산 함수 정의 (cosine distance)
    def cosine_dist(x, y):
        return cosine(x, y)

    # 4. DTW 계산
    alignment = dtw(text_embeds, audio_embeds, dist_method=cosine_dist)

    return alignment


def map_time_code(sentences: list[str], alignment, srt_file_path: str):
    import datetime

    def seconds_to_srt_time(seconds):
        td = datetime.timedelta(seconds=seconds)
        s = str(td)[:-3] if "." in str(td) else str(td) + ".000"
        if len(s.split(":")[0]) == 1:
            s = "0" + s
        return s.replace(".", ",")

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


def test_srt():

    print("✅ SRT 파일이 'output.srt'로 저장되었습니다.")
