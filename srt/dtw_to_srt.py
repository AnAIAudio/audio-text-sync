def run_dtw_to_srt(
    sentences: list[str],
    alignment,
    waveform,
    sample_rate,
    audio_embedding,
    srt_file_path: str = "",
):
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
    duration = waveform.shape[-1] / sample_rate  # 전체 길이(초)
    hidden_len = audio_embedding.shape[0]  # Wav2Vec2 프레임 수
    true_stride = duration / hidden_len  # ✓ 보정된 stride

    frame_to_time = lambda idx: idx * true_stride
    # frame_to_time = lambda idx: idx * 0.02  # 20ms per frame (Wav2Vec2 default stride)

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
        print(f"{idx+1}")
        print(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}")
        print(sentences[idx])
        print()

    with open(srt_file_path, "w", encoding="utf-8") as f:
        for idx, (start, end) in enumerate(sentence_times):
            f.write(f"{idx+1}\n")
            f.write(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}\n")
            f.write(f"{sentences[idx]}\n\n")
