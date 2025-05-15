from audio.cut_wave import (
    stt_using_whisper,
    segment_srt,
    read_srt,
)
from llm.agent import run_agent
from text.text_util import (
    create_text_line,
    SequentialPicker,
    merge_segments,
)
from utils.prepare import (
    prepare_directories,
    test_file_paths,
    read_text_files,
)

if __name__ == "__main__":
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
        formatted,
    ) = test_file_paths(
        audio_directory_path,
        text_directory_path,
        srt_directory_path,
    )

    full_text = read_text_files(text_file_path=text_file_path)
    text_list = create_text_line(raw_text=full_text)

    whisper_result = stt_using_whisper(audio_file_path=audio_file_path)
    segments = whisper_result["segments"]

    original_seq = SequentialPicker(items=text_list)
    merged_segments = merge_segments(
        segments=segments,
        picker=original_seq,
    )

    segment_srt(
        segments=merged_segments,
        srt_file_path=srt_file_path,
    )

    srt = read_srt(srt_file_path=srt_file_path)

    run_agent(
        srt_directory_path=srt_directory_path,
        formatted=formatted,
        text_to_translate=srt,
        language="한국어",
    )
