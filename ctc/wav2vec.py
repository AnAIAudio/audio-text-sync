import os
import time
import torch
import numpy as np
import ctc_segmentation
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
import soundfile as sf
import librosa


def read_text_files(text_file_path):
    # 2. 문장 리스트 준비 (시계열 순서대로 정렬된 문장들)
    with open(text_file_path, "r") as f:
        text = f.read()

    return text


def split_sentences(text: str, language: str = "ko"):
    if language == "ko":
        import kss

        return kss.split_sentences(text, strip=True)
    else:
        import nltk

        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize

        return [s.strip() for s in sent_tokenize(text)]


def create_text_line(raw_text: str, lang_code: str = "en"):
    sentences = []
    for line in raw_text.splitlines():
        if line.strip():  # 빈 줄 건너뛰기
            sentences.extend(split_sentences(line, lang_code))

    return sentences


class Wav2VecModel:
    def __init__(self, transcript: list[str]):
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        self.SAMPLERATE = 16000
        self.TRANSCRIPTS = transcript

    def load_audio(self, file_path: str):
        """오디오 파일을 로드하고 샘플레이트 변환"""
        audio, sr = sf.read(file_path)

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        if sr != self.SAMPLERATE:
            print(f"샘플레이트를 {sr}Hz에서 {self.SAMPLERATE}Hz로 변환합니다.")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLERATE)

        return audio

    def align_with_transcript(
        self,
        audio: np.ndarray,
    ):
        assert audio.ndim == 1
        # 예측 실행, 로짓 및 확률 계산
        inputs = self.processor(
            audio, return_tensors="pt", padding="longest", sampling_rate=self.SAMPLERATE
        )
        # 입력 텐서를 모델과 같은 장치로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(inputs["input_values"]).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].to("cpu")

        # 트랜스크립트 토큰화
        vocab = self.tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        unk_id = vocab["<unk>"]

        tokens = []
        for transcript in self.TRANSCRIPTS:
            assert len(transcript) > 0
            tok_ids = self.tokenizer(transcript.replace("\n", " ").lower())["input_ids"]
            tok_ids = np.array(tok_ids, dtype=int)
            tokens.append(tok_ids[tok_ids != unk_id])

        # 정렬
        char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
        config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
        config.index_duration = audio.shape[0] / probs.size()[0] / self.SAMPLERATE

        ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(
            config, tokens
        )
        timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
            config, probs.numpy(), ground_truth_mat
        )
        segments = ctc_segmentation.determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, self.TRANSCRIPTS
        )
        return [
            {"text": t, "start": p[0], "end": p[1], "conf": p[2]}
            for t, p in zip(self.TRANSCRIPTS, segments)
        ]

    def get_word_timestamps(
        self,
        audio: np.ndarray,
    ):
        assert audio.ndim == 1
        # 예측 실행, 로짓 및 확률 계산
        inputs = self.processor(
            audio, return_tensors="pt", padding="longest", sampling_rate=self.SAMPLERATE
        )
        # 입력 텐서를 모델과 같은 장치로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(inputs["input_values"]).logits
            logits_cpu = logits.cpu()[0]
            probs = torch.nn.functional.softmax(logits_cpu, dim=-1)

        predicted_ids = torch.argmax(logits_cpu, dim=-1)
        pred_transcript = self.processor.decode(predicted_ids)

        # 트랜스크립션을 단어로 나누기
        words = pred_transcript.split(" ")

        # 정렬
        vocab = self.tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
        config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
        config.index_duration = audio.shape[0] / probs.size()[0] / self.SAMPLERATE

        ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(
            config, words
        )
        timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
            config, probs.numpy(), ground_truth_mat
        )
        segments = ctc_segmentation.determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, words
        )
        return [
            {"text": w, "start": p[0], "end": p[1], "conf": p[2]}
            for w, p in zip(words, segments)
        ]

    def write_srt(self, srt_file_path: str, transcript_alignments: list[dict]):
        from datetime import timedelta

        for id, alignment in enumerate(transcript_alignments, start=1):
            start_time = (
                str(0) + str(timedelta(seconds=int(alignment["start"]))) + ",000"
            )
            end_time = str(0) + str(timedelta(seconds=int(alignment["end"]))) + ",000"
            text = alignment["text"]

            if not text:
                continue

            text = text[1:] if text[0] == " " else text
            segment = f"{id}\n{start_time} --> {end_time}\n{text}\n\n"

            with open(srt_file_path, "a", encoding="utf-8") as f:
                f.write(segment)

    def write_timestamp_srt(self, srt_file_path: str, word_timestamps: list[dict]):
        for id, word_timestamp in enumerate(word_timestamps, start=1):
            text = word_timestamp["text"]

            if not text:
                continue

            text = text[1:] if text[0] == " " else text
            if word_timestamp["start"] > word_timestamp["end"]:
                word_timestamp["start"], word_timestamp["end"] = (
                    word_timestamp["end"],
                    word_timestamp["start"],
                )
            segment = f"{id}\n{word_timestamp['start']:.2f} --> {word_timestamp['end']:.2f}\n{text}\n\n"

            with open(srt_file_path, "a", encoding="utf-8") as f:
                f.write(segment)

    def write_timestamp_textgrid(
        self,
        textgrid_file_path: str,
        word_timestamps: list[dict],
        tier_name: str = "words",
    ):
        """word_timestamps 리스트를 TextGrid 형식으로 파일에 작성합니다."""

        start_time = min(item["start"] for item in word_timestamps)
        end_time = max(item["end"] for item in word_timestamps)

        if start_time > end_time:
            start_time, end_time = end_time, start_time

        # TextGrid 헤더 작성
        header = f"""File type = "ooTextFile"
Object class = "TextGrid"

xmin = {start_time:.2f}
xmax = {end_time:.2f}
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "{tier_name}"
        xmin = {start_time:.2f}
        xmax = {end_time:.2f}
        intervals: size = {len(word_timestamps)}
"""

        # 각 간격(interval)에 대한 정보 작성
        intervals = []
        for i, word_timestamp in enumerate(word_timestamps, start=1):
            text = word_timestamp["text"].replace('"', "'")
            if text and text[0] == " ":
                text = text[1:]

            interval = f"""        intervals [{i}]:
                xmin = {word_timestamp["start"]:.2f}
                xmax = {word_timestamp["end"]:.2f}
                text = "{text}"
    """
            intervals.append(interval)

        # 전체 TextGrid 내용 조합
        content = header + "".join(intervals)

        # 파일에 작성
        with open(textgrid_file_path, "w", encoding="utf-8") as f:
            f.write(content)


if __name__ == "__main__":
    print("cuda : ", torch.cuda.is_available())

    start_time = time.time()
    print("start time")
    common_path = os.path.join("audio", "seperated_25min", "NH032")

    if not os.path.exists(common_path):
        os.makedirs(common_path, exist_ok=True)

    text_path = os.path.join(common_path, "NH032.txt")
    full_text = read_text_files(text_file_path=text_path)
    transcript = create_text_line(raw_text=full_text)

    print("init")
    model = Wav2VecModel(transcript=transcript)

    print("load")
    audio_path = os.path.join(common_path, "NH032.mp3")
    audio = model.load_audio(file_path=audio_path)

    print("align")
    transcript_alignments = model.align_with_transcript(audio)
    print("트랜스크립트 정렬 결과:")
    for alignment in transcript_alignments:
        print(f"텍스트: {alignment['text']}")
        print(f"시작: {alignment['start']:.2f}초, 종료: {alignment['end']:.2f}초")
        print("---")

    # model.write_srt(
    #     srt_file_path=srt_file_path,
    #     transcript_alignments=transcript_alignments,
    # )

    # word_timestamps = model.get_word_timestamps(audio)
    # for word_timestamp in word_timestamps:
    #     print(f"텍스트: {word_timestamp['text']}")
    #     print(f"시작: {word_timestamp['start']:.2f}초, 종료: {word_timestamp['end']:.2f}초")
    #     print("---")

    output_dir_path = os.path.join(common_path, "output")
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path, exist_ok=True)

    srt_file_path = os.path.join(output_dir_path, "wav2vec2-large-xlsr-53-english.srt")
    model.write_timestamp_srt(
        srt_file_path=srt_file_path,
        word_timestamps=transcript_alignments,
    )

    textgrid_file_path = os.path.join(
        output_dir_path, "wav2vec2-large-xlsr-53-english.textgrid"
    )
    model.write_timestamp_textgrid(
        textgrid_file_path=textgrid_file_path,
        word_timestamps=transcript_alignments,
    )

    end_time = time.time() - start_time
    print("걸린 시간 : ", end_time)
