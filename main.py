import os

from audio.cut_wave import (
    stt_using_whisper,
    segment_srt,
    write_file,
    read_file,
    write_timestamp_textgrid,
    write_timestamp_srt,
)
from llm.agent import AgentModel
from text.text_util import (
    create_text_line,
    SequentialPicker,
    merge_segments,
)
from utils.prepare import (
    prepare_directories,
    test_file_paths,
    read_json_files,
)
import mfa.alignment

if __name__ == "__main__":
    filename = "NH032"
    mfa_model = "english_us_arpa"
    dataset_directory_path = prepare_directories(filename)
    (
        audio_file_path,
        json_file_path,
        text_file_path,
        srt_file_path,
        textgrid_file_path,
        correct_srt_file_path,
        formatted,
    ) = test_file_paths(dataset_directory_path, filename)

    # whisper 을 통해 stt 결과 받아오는 부분
    # whisper_result = stt_using_whisper(audio_file_path=audio_file_path)
    # whisper_segments = whisper_result["segments"]

    # full_text = " ".join([segment["text"] for segment in whisper_segments])
    # text_list = create_text_line(raw_text=full_text)

    # original_seq = SequentialPicker(items=text_list)
    # merged_segments = merge_segments(
    #     segments=whisper_segments,
    #     picker=original_seq,
    # )
    # write_file(
    #     file_path=text_file_path,
    #     text=" ".join([segment["text"] for segment in merged_segments]),
    # )
    return_code = mfa.alignment.check(model_type="dictionary", model=mfa_model)
    if return_code != 0:
        mfa.alignment.install(
            model_type="dictionary",
            model=mfa_model,
        )

    return_code = mfa.alignment.check(
        model_type="acoustic",
        model=mfa_model,
    )
    if return_code != 0:
        mfa.alignment.install(
            model_type="acoustic",
            model=mfa_model,
        )

    mfa.alignment.run(
        data_path=dataset_directory_path,
        mfa_model=mfa_model,
        output_path=dataset_directory_path,
    )

    aligend_texts = read_json_files(json_file_path)
    full_text = read_file(text_file_path)
    text_list = create_text_line(full_text)

    merged_segments = mfa.alignment.merge(
        aligned_texts=aligend_texts["tiers"]["words"]["entries"],
        text_list=text_list,
    )
    write_timestamp_textgrid(
        textgrid_file_path=textgrid_file_path,
        word_timestamps=merged_segments,
    )
    write_timestamp_srt(
        srt_file_path=srt_file_path,
        word_timestamps=merged_segments,
    )

    # Agent 을 이용해 번역하는 부분
    # agent = AgentModel()  # system_prompt, compare_system_prompt 여기서 변경 가능
    # agent.run(
    #     srt_directory_path=srt_file_path,
    #     formatted=formatted,
    #     segments=merged_segments,
    #     language="한국어",
    #     seperate_number=50,
    # )
