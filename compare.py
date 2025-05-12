def compare_dtw(text_embeds, audio_embeds):
    from dtw import dtw
    from scipy.spatial.distance import cosine

    # 3. 거리 계산 함수 정의 (cosine distance)
    def cosine_dist(x, y):
        return cosine(x, y)

    # 4. DTW 계산
    alignment = dtw(text_embeds, audio_embeds, dist=cosine_dist)

    return alignment


def map_time_code(sentences: list[str], alignment):
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

    # 3. SRT 형식 출력
    print("\nGenerated SRT:")
    for idx, (start, end) in enumerate(sentence_times):
        print(f"{idx + 1}")
        print(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}")
        print(sentences[idx])
        print()
