from datetime import datetime
import torch
from audio.wave_to_vector import run_wave2vec
from compare import compare_dtw, map_time_code
from text.bert import run_bert

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

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run_mfcc(audio_file_path=audio_file_path)
    # run_tdf(text_file_path=text_file_path)

    # 2. 문장 리스트 준비 (시계열 순서대로 정렬된 문장들)
    with open(text_file_path, "r") as f:
        text_data = f.readlines()

    ss = run_bert(text_data=text_data)
    zz = run_wave2vec(audio_file_path=audio_file_path)
    gg = compare_dtw(ss, zz)

    now = datetime.now()
    formatted = now.strftime("%Y%m%d%H%M%S")
    srt_file_path = os.path.join(SRT_DIRECTORY_PATH, f"voix_result_srt_{formatted}.srt")

    map_time_code(
        sentences=text_data,
        alignment=gg,
        srt_file_path=srt_file_path,
    )
