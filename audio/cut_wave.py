from typing import List
from whisper import Whisper

from text.text_util import Segment


def load_whisper_model() -> Whisper:
    """
    위 함수는 Whisper 모델을 불러오는 기능을 담당합니다. 사용자가 지정한 모델명이
    사용 가능한 모델 목록에 없는 경우 기본 모델인 'base'를 사용하여 모델을 로드합니다.
    결과적으로 로드된 모델 객체를 반환하게 됩니다.

    Returns:
        모델 객체를 반환합니다. 기본 제공 모델 이름 목록에서 모델을 로드합니다.
    """
    import whisper

    # model_name = "large-v3"
    model_name = "medium"

    available_model_list = whisper.available_models()

    if model_name not in available_model_list:
        model_name = "base"

    model = whisper.load_model(model_name)
    return model


def stt_using_whisper(audio_file_path: str = None, target_lang: str = "en"):
    """
    Whisper 모델을 사용해서 STT 생성

    Args:
        audio_file_path: 오디오 파일 경로
        target_lang: ISO 규격을 따르는 언어 줄임말

    Returns:
        TimeStamp와 텍스트가 포함된 객체 목록
    """

    model = load_whisper_model()

    if not audio_file_path:
        return []

    transcribe_result = model.transcribe(
        audio=audio_file_path,
        verbose=True,
        language=target_lang,
    )

    return transcribe_result


def format_to_srt_time(timestamp):
    """
    초 단위의 시간을 SRT 포맷(HH:MM:SS,mmm)으로 변환합니다.

    :param timestamp: 초 단위의 시간(float)
    :return: SRT 형식의 시간 문자열
    """
    milliseconds = int(timestamp * 1000)
    hours = milliseconds // (1000 * 60 * 60)
    minutes = (milliseconds % (1000 * 60 * 60)) // (1000 * 60)
    seconds = (milliseconds % (1000 * 60)) // 1000
    milliseconds = milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def segment_srt(segments: List[Segment], srt_file_path: str):
    from datetime import timedelta

    # text_list = []
    for id, segment in enumerate(segments, start=1):
        start_time = str(0) + str(timedelta(seconds=int(segment["start"]))) + ",000"
        end_time = str(0) + str(timedelta(seconds=int(segment["end"]))) + ",000"
        text = segment["text"]
        # segment_id = segment["id"] + 1
        if not text:
            continue

        text = text[1:] if text[0] == " " else text
        segment = f"{id}\n{start_time} --> {end_time}\n{text}\n\n"
        # text_list.append(text)

        with open(srt_file_path, "a", encoding="utf-8") as f:
            f.write(segment)


def write_timestamp_srt(srt_file_path: str, word_timestamps: list[dict]):
    for id, word_timestamp in enumerate(word_timestamps, start=1):
        text = word_timestamp["text"]

        if not text:
            continue

        text = text[1:] if text[0] == " " else text
        if word_timestamp["start"] > word_timestamp["end"]:
            word_timestamp["start"], word_timestamp["end"] = (
                word_timestamp["end"],
                word_timestamp["start"],
            )
        segment = f"{id}\n{format_to_srt_time(word_timestamp['start'])} --> {format_to_srt_time(word_timestamp['end'])}\n{text}\n\n"

        with open(srt_file_path, "a", encoding="utf-8") as f:
            f.write(segment)


def write_timestamp_textgrid(
    textgrid_file_path: str,
    word_timestamps: list[dict],
    tier_name: str = "words",
):
    """word_timestamps 리스트를 TextGrid 형식으로 파일에 작성합니다."""

    start_time = min(item["start"] for item in word_timestamps)
    end_time = max(item["end"] for item in word_timestamps)

    if start_time > end_time:
        start_time, end_time = end_time, start_time

    # TextGrid 헤더 작성
    header = f"""File type = "ooTextFile"
Object class = "TextGrid"

xmin = {start_time:.2f}
xmax = {end_time:.2f}
tiers? <exists>
size = 1
item []:
item [1]:
    class = "IntervalTier"
    name = "{tier_name}"
    xmin = {start_time:.2f}
    xmax = {end_time:.2f}
    intervals: size = {len(word_timestamps)}
"""

    # 각 간격(interval)에 대한 정보 작성
    intervals = []
    for i, word_timestamp in enumerate(word_timestamps, start=1):
        text = word_timestamp["text"].replace('"', "'")
        if text and text[0] == " ":
            text = text[1:]

        interval = f"""        intervals [{i}]:
            xmin = {word_timestamp["start"]:.2f}
            xmax = {word_timestamp["end"]:.2f}
            text = "{text}"
"""
        intervals.append(interval)

    # 전체 TextGrid 내용 조합
    content = header + "".join(intervals)

    # 파일에 작성
    with open(textgrid_file_path, "w", encoding="utf-8") as f:
        f.write(content)


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(file_path: str, text: str):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text)
