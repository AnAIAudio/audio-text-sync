import os

from audio.cut_wave import (
    stt_using_whisper,
    segment_srt,
    write_file, write_timestamp_textgrid, write_timestamp_srt,
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
from bert.sentence import bert_seperate

if __name__ == "__main__":
    dataset_directory_path = prepare_directories()
    (
        audio_file_path,
        json_file_path,
        text_file_path,
        srt_file_path,
        textgrid_file_path,
        correct_srt_file_path,
        formatted,
    ) = test_file_paths(dataset_directory_path)

    # whisper_result = stt_using_whisper(audio_file_path=audio_file_path)
    # whisper_segments = whisper_result["segments"]
    #
    # full_text = " ".join([segment["text"] for segment in whisper_segments])
    # text_list = create_text_line(raw_text=full_text)
    #
    # original_seq = SequentialPicker(items=text_list)
    # merged_segments = merge_segments(
    #     segments=whisper_segments,
    #     picker=original_seq,
    # )
    # write_file(
    #     file_path=text_file_path,
    #     text=" ".join([segment["text"] for segment in merged_segments]),
    # )
    #
    # mfa.alignment.run(
    #     data_path=dataset_directory_path,
    #     dict_path="english_us_arpa",
    #     acoustic_path="english_us_arpa",
    #     output_path=dataset_directory_path,
    #     config_path=os.path.join("mfa", "config.yaml"),
    # )
    #
    # aligend_texts = read_json_files(json_file_path)
    #
    # # MFA 에서 단어 단위로 잘라진 내용을 문장으로 만들기 위해 whisper 에서 반환한 text 를 이용( whisper 가 문장 분리를 잘 했다는 전제가 필요함 )
    # idx = 0  # MFA 인덱스
    # max_length = len(aligend_texts["tiers"]["words"]["entries"])
    # for segment in merged_segments:
    #     text = segment["text"]
    #     if aligend_texts["tiers"]["words"]["entries"][idx][2] in text:
    #         segment["start"] = aligend_texts["tiers"]["words"]["entries"][idx][0]
    #
    #     while (
    #         idx < max_length
    #         and aligend_texts["tiers"]["words"]["entries"][idx][2] in text
    #     ):
    #         aligend_texts["tiers"]["words"]["entries"][idx][2] = text.replace(
    #             aligend_texts["tiers"]["words"]["entries"][idx][2], "", 1
    #         )
    #         segment["end"] = aligend_texts["tiers"]["words"]["entries"][idx][1]
    #         idx += 1
    #
    # print(merged_segments)
    #
    # segment_srt(
    #     segments=whisper_segments,
    #     srt_file_path=srt_file_path,
    # )
    #
    # agent = AgentModel()  # system_prompt, compare_system_prompt 여기서 변경 가능
    # agent.run(
    #     srt_directory_path=srt_file_path,
    #     formatted=formatted,
    #     segments=merged_segments,
    #     language="한국어",
    #     seperate_number=50,
    # )

    aligend_texts = read_json_files(json_file_path)
    mfa_segments = [
        {"start": entry[0], "end": entry[1], "text": entry[2]}
        for entry in aligend_texts["tiers"]["words"]["entries"]
    ]

    merged_mfa_segments = bert_seperate(mfa_segments)
    write_timestamp_textgrid(
        textgrid_file_path=textgrid_file_path,
        word_timestamps=merged_mfa_segments,
    )
    write_timestamp_srt(
        srt_file_path=srt_file_path,
        word_timestamps=merged_mfa_segments,
    )
