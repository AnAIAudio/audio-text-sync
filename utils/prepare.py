import json


def prepare_directories():
    import os
    from custom_path import MAIN_BASE_PATH

    dataset_directory_path = os.path.join(MAIN_BASE_PATH, "datasets", "S23")

    path_list = [
        dataset_directory_path,
    ]

    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    return dataset_directory_path


def test_file_paths(datasets_directory_path: str):
    import os
    from datetime import datetime

    audio_file_path = os.path.join(datasets_directory_path, "S23.mp3")
    json_file_path = os.path.join(datasets_directory_path, "S23.json")
    text_file_path = os.path.join(datasets_directory_path, f"S23.txt")
    now = datetime.now()
    formatted = now.strftime("%Y%m%d%H%M%S")
    srt_file_path = os.path.join(datasets_directory_path, f"S23_{formatted}.srt")
    textgrid_file_path = os.path.join(datasets_directory_path, f"S23_{formatted}.textgrid")
    correct_srt_file_path = os.path.join(
        datasets_directory_path, f"S23_{formatted}.srt"
    )

    return (
        audio_file_path,
        json_file_path,
        text_file_path,
        srt_file_path,
        textgrid_file_path,
        correct_srt_file_path,
        formatted,
    )


def read_text_files(text_file_path):
    # 2. 문장 리스트 준비 (시계열 순서대로 정렬된 문장들)
    with open(text_file_path, "r") as f:
        text = f.read()

    return text


def read_json_files(json_file_path):
    with open(json_file_path, "r") as f:
        text = json.load(f)
    return text
