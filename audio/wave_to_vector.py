import os


def run_wave2vec(audio_file_path: str):
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # pip install torchaudio transformers

    import torch
    import torchaudio
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    # 모델 로딩
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    # 오디오 로드 (16kHz mono)
    waveform, sample_rate = torchaudio.load(audio_file_path)

    # 스테레오 → 단일 채널 변환
    # waveform = waveform.squeeze()  # (1, T) → (T,)
    if waveform.shape[0] > 1:
        # 여러 채널이 있을 경우 평균 내서 단일 채널로
        waveform = waveform.mean(dim=0)
    else:
        # 단일 채널인 경우 (1, T) → (T,)
        waveform = waveform.squeeze()

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        waveform = resampler(waveform)

    # 입력 준비
    input_values = processor(
        waveform, sampling_rate=16000, return_tensors="pt"
    ).input_values

    # 임베딩 추출
    with torch.no_grad():
        embeddings = model(input_values).last_hidden_state.squeeze(0)  # (T, D)

    print(embeddings.shape)  # 예: (500, 768)

    return embeddings
