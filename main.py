import torch
from audio.cut_wave import (
    stt_using_whisper,
    whisper_text,
    whisper_srt,
    segment_srt,
)
from text.text_util import (
    create_text_line,
    split_sentences,
    SequentialPicker,
    merge_segments,
    is_complete,
)
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
    segments = whisper_result["segments"]
    whisper_text_list = whisper_text(whisper_result["segments"])

    zz = SequentialPicker(items=text_list)

    test = merge_segments(segments=whisper_result["segments"], picker=zz)

    picked_text_list = []
    temp_text = ""
    # for whisper_text in whisper_text_list:
    #     if not is_complete(whisper_text):
    #         temp_text += whisper_text
    #         continue
    #
    #     dddd = split_sentences(text=temp_text)
    #     zzzz = len(dddd)
    #     temp_text = ""
    #
    #     picker_list = zz.take(n=zzzz)
    #     # picker_list를 srt에 whisper text 대신 넣어야 함
    #     picked_text_list.extend(picker_list)

    segment_srt(
        segments=test,
        srt_file_path=srt_file_path,
    )

    # whisper_srt(
    #     segments=whisper_result["segments"],
    #     text_list=picked_text_list,
    #     srt_file_path=srt_file_path,
    # )
