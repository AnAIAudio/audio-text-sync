from datetime import datetime
import torch
from sklearn.preprocessing import normalize
from audio.wave_to_vector import run_wave2vec
from compare import compare_dtw, seconds_to_srt_time, run_dtw
from srt.dtw_to_srt import run_dtw_to_srt
from text.sentence_bert import run_sentence_bert
from text.text_util import create_text_line_full_text
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

    full_text = read_text_files(text_file_path=text_file_path)
    text_list = create_text_line_full_text(raw_text=full_text)

    text_embedding = run_sentence_bert(text_list=text_list)
    audio_embedding = run_wave2vec(audio_file_path=audio_file_path)

    alignment = run_dtw(text_embedding, audio_embedding, srt_file_path)

    run_dtw_to_srt(
        sentences=text_list,
        alignment=alignment,
        srt_file_path=srt_file_path,
    )
