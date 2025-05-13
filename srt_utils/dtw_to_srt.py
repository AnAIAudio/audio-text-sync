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

    duration = waveform.shape[-1] / sample_rate
    hidden_len = audio_embedding.shape[0]
    stride = duration / hidden_len  # 반드시 실측으로 구하기

    # DTW 매핑 후
    # mapped_frames_per_sentence = [[] for _ in range(n_sentences)]
    # for sent_idx, frame_idx in zip(alignment.index1, alignment.index2):
    #     mapped_frames_per_sentence[sent_idx].append(frame_idx)

    mapped_frames_per_sentence: list[list[int]] = [[] for _ in range(len(sentences))]
    for sent_idx, frame_idx in zip(alignment.index1, alignment.index2):
        if sent_idx >= len(mapped_frames_per_sentence):
            # 예외 상황: 문장 수보다 큰 ID가 들어옴 → 확장 & 경고
            extra = sent_idx + 1 - len(mapped_frames_per_sentence)
            mapped_frames_per_sentence.extend([[] for _ in range(extra)])
            print(
                f"[Warn] sent_idx {sent_idx} exceeds sentence list; list auto-expanded."
            )
        mapped_frames_per_sentence[sent_idx].append(int(frame_idx))

    # sentence_times = []
    # for frames in mapped_frames_per_sentence:
    #     if not frames:
    #         continue
    #     ss, ee = np.percentile(frames, [10, 90])  # ③ 꼬리 자르기
    #     start_sec = ss * stride
    #     end_sec = ee * stride
    #     # 최소 길이 보장 & 앞뒤 겹침 방지 루틴 ...
    #     sentence_times.append((start_sec, end_sec))

    min_dur: float = 0.5  # 최소 0.5 s
    gap: float = 0.1  # 문장 간 최소 간격

    sentence_times: list[tuple[float, float] | None] = [None] * len(
        mapped_frames_per_sentence
    )
    for i, frames in enumerate(mapped_frames_per_sentence):
        if not frames:
            continue  # 매핑 안 된 문장은 None
        ss, ee = np.percentile(frames, [10, 90])  # 꼬리 자르기
        start = ss * stride
        end = ee * stride
        if end - start < min_dur:  # 최소 길이 보장
            pad = (min_dur - (end - start)) / 2
            start -= pad
            end += pad
        sentence_times[i] = (max(0, start), min(duration, end))

    # ── 3. 앞뒤 겹침 보정
    last_end = 0.0
    for i, t in enumerate(sentence_times):
        if t is None:
            continue
        st, ed = t
        if st < last_end + gap:  # 겹치면 밀어내기
            shift = last_end + gap - st
            st += shift
            ed += shift
            sentence_times[i] = (st, min(ed, duration))
        last_end = sentence_times[i][1]

    # 3. SRT 형식 출력
    lines = []
    idx = 1
    for i, t in enumerate(sentence_times):
        if t is None:
            continue  # 매핑 실패 문장 skip
        if i >= len(sentences):  # 텍스트 없으면 placeholder
            sent_text = f"<missing-{i}>"
        else:
            sent_text = sentences[i]

        st, ed = t
        lines.append(f"{idx}")
        lines.append(f"{seconds_to_srt_time(st)} --> {seconds_to_srt_time(ed)}")
        lines.append(sent_text)
        lines.append("")
        idx += 1

    # Path(srt_file_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"✅  SRT saved: {srt_file_path}")

    with open(srt_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        # for idx, (start, end) in enumerate(sentence_times):
        #     f.write(f"{idx+1}\n")
        #     f.write(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}\n")
        #     f.write(f"{sentences[idx]}\n\n")
