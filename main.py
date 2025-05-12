from datetime import datetime
import torch
from sklearn.preprocessing import normalize
from audio.wave_to_vector import run_wave2vec
from compare import compare_dtw, map_time_code, seconds_to_srt_time
from text.bert import run_bert
from text.word_bert import run_visualize, run_word_bert

if __name__ == "__main__":
    import os
    from custom_path import MAIN_BASE_PATH

    TEMP_DIRECTORY_PATH = os.path.join(MAIN_BASE_PATH, "temp")
    AUDIO_DIRECTORY_PATH = os.path.join(MAIN_BASE_PATH, "audio")
    TEXT_DIRECTORY_PATH = os.path.join(MAIN_BASE_PATH, "text")
    SRT_DIRECTORY_PATH = os.path.join(MAIN_BASE_PATH, "srt")

    path_list = [
        TEMP_DIRECTORY_PATH,
        AUDIO_DIRECTORY_PATH,
        TEXT_DIRECTORY_PATH,
        SRT_DIRECTORY_PATH,
    ]

    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    audio_file_path = os.path.join(AUDIO_DIRECTORY_PATH, "voix_result_mp3.mp3")
    text_file_path = os.path.join(TEXT_DIRECTORY_PATH, "voix_result_txt.txt")
    now = datetime.now()
    formatted = now.strftime("%Y%m%d%H%M%S")
    srt_file_path = os.path.join(SRT_DIRECTORY_PATH, f"voix_result_srt_{formatted}.srt")

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 문장 리스트 준비 (시계열 순서대로 정렬된 문장들)
    with open(text_file_path, "r") as f:
        text_data = f.readlines()

    # text_bert = run_bert(text_data=text_data)
    tokens, token_embeddings, token_sentence_ids = run_word_bert(
        sentences=text_data,
        full_text="\n".join(text_data),
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
