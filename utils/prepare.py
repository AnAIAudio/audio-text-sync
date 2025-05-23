import json


def prepare_directories(filename: str):
    import os
    from custom_path import MAIN_BASE_PATH

    dataset_directory_path = os.path.join(MAIN_BASE_PATH, "datasets", filename)
    mfa_directory_path = os.path.join(MAIN_BASE_PATH, "mfa")

    path_list = [
        dataset_directory_path,
        mfa_directory_path,
    ]

    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    return (dataset_directory_path, mfa_directory_path)


def test_file_paths(
    datasets_directory_path: str, mfa_directory_path: str, filename: str
):
    import os
    from datetime import datetime

    audio_file_path = os.path.join(datasets_directory_path, f"{filename}.mp3")
    json_file_path = os.path.join(datasets_directory_path, f"{filename}.json")
    text_file_path = os.path.join(datasets_directory_path, f"{filename}.txt")
    now = datetime.now()
    formatted = now.strftime("%Y%m%d%H%M%S")
    srt_file_path = os.path.join(datasets_directory_path, f"{filename}_{formatted}.srt")
    textgrid_file_path = os.path.join(
        datasets_directory_path, f"{filename}_{formatted}.textgrid"
    )
    correct_srt_file_path = os.path.join(
        datasets_directory_path, f"{filename}_{formatted}.srt"
    )
    mfa_acoustic_path = os.path.join(
        mfa_directory_path, "acoustic", "english_us_arpa.zip"
    )
    mfa_dict_path = os.path.join(
        mfa_directory_path, "dictionary", "english_us_arpa.dict"
    )

    return (
        audio_file_path,
        json_file_path,
        text_file_path,
        srt_file_path,
        textgrid_file_path,
        correct_srt_file_path,
        mfa_acoustic_path,
        mfa_dict_path,
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
