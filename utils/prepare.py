def prepare_directories():
    import os
    from custom_path import MAIN_BASE_PATH

    temp_directory_path = os.path.join(MAIN_BASE_PATH, "temp")
    audio_directory_path = os.path.join(MAIN_BASE_PATH, "audio")
    text_directory_path = os.path.join(MAIN_BASE_PATH, "text")
    srt_directory_path = os.path.join(MAIN_BASE_PATH, "srt_utils")

    path_list = [
        temp_directory_path,
        audio_directory_path,
        text_directory_path,
        srt_directory_path,
    ]

    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    return (
        temp_directory_path,
        audio_directory_path,
        text_directory_path,
        srt_directory_path,
    )


def test_file_paths(audio_directory_path, text_directory_path, srt_directory_path):
    import os
    from datetime import datetime
    from custom_path import MAIN_BASE_PATH

    audio_file_path = os.path.join(audio_directory_path, "voix_result_mp3.mp3")
    text_file_path = os.path.join(text_directory_path, "voix_result_txt.txt")
    now = datetime.now()
    formatted = now.strftime("%Y%m%d%H%M%S")
    srt_file_path = os.path.join(srt_directory_path, f"voix_result_srt_{formatted}.srt_utils")
    correct_srt_file_path = os.path.join(srt_directory_path, "correct_srt.srt_utils")

    return audio_file_path, text_file_path, srt_file_path, correct_srt_file_path


def read_text_files(text_file_path):
    # 2. 문장 리스트 준비 (시계열 순서대로 정렬된 문장들)
    with open(text_file_path, "r") as f:
        text = f.read()

    return text
