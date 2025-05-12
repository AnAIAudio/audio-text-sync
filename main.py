from datetime import datetime
import torch
from sklearn.preprocessing import normalize
from audio.wave_to_vector import run_wave2vec
from compare import compare_dtw, map_time_code, seconds_to_srt_time
from text.bert import run_bert
from text.word_bert import run_visualize, run_word_bert
from utils.prepare import (
    prepare_directories,
    test_file_paths,
    read_text_files,
)

if __name__ == "__main__":
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        temp_directory_path,
        audio_directory_path,
        text_directory_path,
        srt_directory_path,
    ) = prepare_directories()
    audio_file_path, text_file_path, srt_file_path = test_file_paths(
        audio_directory_path,
        text_directory_path,
        srt_directory_path,
    )

    text_list, full_text = read_text_files(text_file_path=text_file_path)

    # text_bert = run_bert(text_data=text_data)
    tokens, token_embeddings, token_sentence_ids = run_word_bert(
        sentences=text_list,
        full_text=full_text,
    )
    audio_embeddings = run_wave2vec(audio_file_path=audio_file_path)

    alignment = compare_dtw(token_embeddings, audio_embeddings)
    mapping = list(zip(alignment.index1, alignment.index2))  # (token_idx, audio_idx)

    # 9. 토큰별 오디오 프레임 수집
    token_frames = {i: [] for i in range(len(tokens))}
    for i, j in mapping:
        token_frames[i].append(j)

    # 10. 토큰 → 문장 단위 그룹화 및 SRT 구성
    srt_entries = []
    for idx in range(len(tokens)):
        frames = token_frames[idx]
        if not frames:
            continue
        start_time = min(frames) * 0.02
        end_time = max(frames) * 0.02 + 0.3
        srt_entries.append(
            (
                idx + 1,
                seconds_to_srt_time(start_time),
                seconds_to_srt_time(end_time),
                tokens[idx],
                token_sentence_ids[idx],
            )
        )

    # 문장 기준 그룹화
    grouped_srt = {}
    for _, start, end, token, sid in srt_entries:
        grouped_srt.setdefault(sid, []).append((start, end, token))

    # 11. SRT 포맷 출력
    with open(srt_file_path, "w", encoding="utf-8") as f:
        srt_index = 1
        for sid, group in grouped_srt.items():
            start = group[0][0]
            end = group[-1][1]
            text = " ".join(tok for _, _, tok in group)
            f.write(f"{srt_index}\n{start} --> {end}\n{text}\n\n")
            srt_index += 1

    run_visualize(alignment=alignment)
