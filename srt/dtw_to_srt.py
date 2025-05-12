def run_dtw_to_srt(
    sentences: list[str],
    alignment,
    waveform,
    sample_rate,
    audio_embedding,
    srt_file_path: str = "",
):
    import numpy as np
    from compare import seconds_to_srt_time

    # 1. 문장 개수
    n_sentences = len(sentences)

    # 2. 문장별로 매핑된 오디오 프레임들 추출
    duration = waveform.shape[-1] / sample_rate  # 전체 길이(초)
    hidden_len = audio_embedding.shape[0]  # Wav2Vec2 프레임 수
    true_stride = duration / hidden_len  # ✓ 보정된 stride
    # frame_to_time = lambda idx: idx * 0.02  # 20ms per frame (Wav2Vec2 default stride)
    frame_to_time = lambda idx: idx * true_stride

    # DTW 매핑 후
    mapped_frames_per_sentence = [[] for _ in range(n_sentences)]
    for sent_idx, frame_idx in zip(alignment.index1, alignment.index2):
        mapped_frames_per_sentence[sent_idx].append(frame_idx)

    sentence_times = []
    for frames in mapped_frames_per_sentence:
        if not frames:
            continue
        ss, ee = np.percentile(frames, [10, 90])  # ③ 꼬리 자르기
        start_sec = ss * true_stride
        end_sec = ee * true_stride
        # 최소 길이 보장 & 앞뒤 겹침 방지 루틴 ...
        sentence_times.append((start_sec, end_sec))

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
