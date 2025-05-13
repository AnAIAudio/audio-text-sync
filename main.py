from datetime import datetime
import torch
from sklearn.preprocessing import normalize
from audio.wave_to_vector import run_wave2vec
from compare import compare_dtw, seconds_to_srt_time, run_dtw
from srt_utils.dtw_to_srt import run_dtw_to_srt
from text.sentence_bert import run_sentence_bert, run_token_level_bert
from text.text_util import create_text_line
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

    # text_embedding = run_sentence_bert(text_list=text_list)
    text_embedding = run_token_level_bert(texts=text_list, device="cpu")
    audio_embedding, waveform, sample_rate = run_wave2vec(
        audio_file_path=audio_file_path
    )

    alignment = run_dtw(text_embedding, audio_embedding)
    # alignment = compare_dtw(text_embedding, audio_embedding)

    run_dtw_to_srt(
        sentences=text_list,
        alignment=alignment,
        srt_file_path=srt_file_path,
        waveform=waveform,
        sample_rate=sample_rate,
        audio_embedding=audio_embedding,
    )

    visualize(
        alignment=alignment,
        audio_path=audio_file_path,
        correct_srt_path=correct_srt_file_path,
        created_srt_path=srt_file_path,
    )
