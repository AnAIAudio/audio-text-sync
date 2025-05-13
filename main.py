from datetime import datetime
import torch
from sklearn.preprocessing import normalize
import os
from audio.cut_wave import (
    stt_using_whisper,
    whisper_text,
    whisper_srt,
    cut_wav,
    check_audio_sync,
)
from audio.wave_to_vector import run_wave2vec, audio_embed_list
from compare import compare_dtw, seconds_to_srt_time, run_dtw
from srt_utils.dtw_to_srt import run_dtw_to_srt
from text.sentence_bert import run_sentence_bert, run_token_level_bert
from text.text_util import create_text_line, split_sentences, SequentialPicker
from text.word_bert import run_visualize, run_word_bert
from utils.prepare import (
    prepare_directories,
    test_file_paths,
    read_text_files,
)
from utils.visualsize import visualize

if __name__ == "__main__":
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        temp_directory_path,
        audio_directory_path,
        text_directory_path,
        srt_directory_path,
    ) = prepare_directories()
    (
        audio_file_path,
        text_file_path,
        srt_file_path,
        correct_srt_file_path,
    ) = test_file_paths(
        audio_directory_path,
        text_directory_path,
        srt_directory_path,
    )

    full_text = read_text_files(text_file_path=text_file_path)
    text_list = create_text_line(raw_text=full_text)

    whisper_result = stt_using_whisper(audio_file_path=audio_file_path)
    whisper_text_list = whisper_text(whisper_result["segments"])

    zz = SequentialPicker(items=text_list)
    picked_text_list = []
    for whisper_text in whisper_text_list:
        dddd = split_sentences(text=whisper_text)
        zzzz = len(dddd)

        picker_list = zz.take(n=zzzz)
        picked_text_list.extend(picker_list)
        # picker_list를 srt에 whisper text 대신 넣어야 함

    whisper_srt(
        segments=whisper_result["segments"],
        text_list=picked_text_list,
        srt_file_path=srt_file_path,
    )
