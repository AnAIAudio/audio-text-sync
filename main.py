from datetime import datetime
import torch
from sklearn.preprocessing import normalize
from audio.wave_to_vector import run_wave2vec
from compare import compare_dtw, map_time_code, seconds_to_srt_time, run_dtw
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

    run_sentence_bert(text_list=text_list)
    run_wave2vec(audio_file_path=audio_file_path)

    run_dtw()
