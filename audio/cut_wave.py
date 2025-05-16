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

    model_name = "large-v3"
    # model_name = "medium"

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


def read_srt(srt_file_path: str) -> str:
    with open(srt_file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_srt(srt_file_path: str, text: str):
    with open(srt_file_path, "w", encoding="utf-8") as f:
        f.write(text)
