import torch
import numpy as np
import ctc_segmentation
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
import soundfile as sf
import librosa

class Wav2VecModel:
    def __init__(self, transcript: list[str]):
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, trust_remote_code=True)
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
        inputs = self.processor(audio, return_tensors="pt", padding="longest", sampling_rate=self.SAMPLERATE)
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits.cpu()[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)

        # 트랜스크립트 토큰화
        vocab = self.tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        unk_id = vocab["<unk>"]

        tokens = []
        for transcript in self.TRANSCRIPTS:
            assert len(transcript) > 0
            tok_ids = self.tokenizer(transcript.replace("\n", " ").lower())['input_ids']
            tok_ids = np.array(tok_ids, dtype=int)
            tokens.append(tok_ids[tok_ids != unk_id])

        # 정렬
        char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
        config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
        config.index_duration = audio.shape[0] / probs.size()[0] / self.SAMPLERATE

        ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokens)
        timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
        segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings,
                                                                 self.TRANSCRIPTS)
        return [{"text": t, "start": p[0], "end": p[1], "conf": p[2]} for t, p in zip(self.TRANSCRIPTS, segments)]


if __name__ == "__main__":
    transcript = """
        This is a pound of Jell-O, (gentle music) but this is 30,000 pounds of Jell-O.
        And I made it 'cause ever since I was a kid, I've always wondered what it would look like to belly flop into the world's largest pool of Jell-O.
        I started by digging a hole in my brother's backyard, but soon realized I needed a completely new way of making Jell-O for this to work, 'cause if I made it the normal way where you boil water on your stove, then mix in the powder, then refrigerate it for it to actually get firm, it would take 3,000 batches and three months to pull off.
        So to scale things up, I took six 55-gallon drums with a custom welded spigot and a custom propane burner stand.
        Then I filled each drum with water, gelatin powder, and red food coloring, and once it boiled, released it layer by layer into the pool every day for seven days.
        Now, as for the refrigeration, I teamed up with Mother Nature, 'cause after studying the weather almanac for his city, I chose the exact three-week window in the year where the weather could cool the Jell-O to the perfect fridge temperature each night without freezing it.
        It was so much freaking hard work to pull this off, but I was so stoked it finally worked, 'cause it turns out belly flopping on a Jell-O pool is actually way cooler than I'd always imagined.
        """.strip().split('\n')
    model = Wav2VecModel(transcript=transcript)

    audio = model.load_audio(file_path="audio/voix_result_mp3.mp3")

    transcript_alignments = model.align_with_transcript(audio)
    print("트랜스크립트 정렬 결과:")
    for alignment in transcript_alignments:
        print(f"텍스트: {alignment['text']}")
        print(f"시작: {alignment['start']:.2f}초, 종료: {alignment['end']:.2f}초")
        print("---")
