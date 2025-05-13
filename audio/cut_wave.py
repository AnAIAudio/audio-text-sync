import subprocess, os
from pathlib import Path
import srt, datetime
import numpy as np
from whisper import Whisper


def cut_wav(audio_path, srt_path, out_dir):
    Path(out_dir).mkdir(exist_ok=True)
    srt_txt = Path(srt_path).read_text(encoding="utf-8")
    subs = list(srt.parse(srt_txt))
    for i, sub in enumerate(subs, 1):
        s, e = sub.start.total_seconds(), sub.end.total_seconds()
        out = Path(out_dir) / f"sent_{i:03d}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-ss",
                f"{s}",
                "-to",
                f"{e}",
                "-c",
                "copy",
                str(out),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


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
    model_name = "base"

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


def whisper_text(segments):
    text_list = []
    for segment in segments:
        text = segment["text"]
        text_list.append(text)
    return text_list


def whisper_srt(segments, text_list: list[str], srt_file_path: str):
    from datetime import timedelta

    # text_list = []
    # for segment in segments:
    #     start_time = str(0) + str(timedelta(seconds=int(segment["start"]))) + ",000"
    #     end_time = str(0) + str(timedelta(seconds=int(segment["end"]))) + ",000"
    #     text = segment["text"]
    #     segment_id = segment["id"] + 1
    #     segment = f"{segment_id}\n{start_time} --> {end_time}\n{text[1:] if text[0] is ' ' else text}\n\n"
    #     # text_list.append(text)
    #
    #     with open(srt_file_path, "a", encoding="utf-8") as f:
    #         f.write(segment)

    for i, text in enumerate(text_list, 1):
        # print(f"{i:03d}  {text}")

        # start_time = str(0) + str(timedelta(seconds=int(segment["start"]))) + ",000"
        start_time = str(0) + str(timedelta(seconds=int(segments[i]["start"]))) + ",000"

        # end_time = str(0) + str(timedelta(seconds=int(segment["end"]))) + ",000"
        end_time = str(0) + str(timedelta(seconds=int(segments[i]["end"]))) + ",000"
        # text = segment["text"]

        segments[i]["text"] = text

        segment_id = segments[i]["id"] + 1
        segment = f"{segment_id}\n{start_time} --> {end_time}\n{text[1:] if text[0] is ' ' else text}\n\n"
        # text_list.append(text)

        with open(srt_file_path, "a", encoding="utf-8") as f:
            f.write(segment)

    # return text_list


def check_audio_sync(sent_embeds, audio_embeds):
    from scipy.spatial.distance import cosine

    for i, (t_vec, a_vec) in enumerate(zip(sent_embeds, audio_embeds), 1):
        sim = 1 - cosine(t_vec, a_vec)
        print(f"{i:03d}  cos_sim = {sim:.3f}")
