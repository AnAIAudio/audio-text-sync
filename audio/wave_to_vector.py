def run_wave2vec():
    # pip install torchaudio transformers

    import torch
    import torchaudio
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    # 모델 로딩
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    # 오디오 로드 (16kHz mono)
    waveform, sample_rate = torchaudio.load("your_audio.wav")
    waveform = waveform.squeeze()  # (1, T) → (T,)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # 입력 준비
    input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values

    # 임베딩 추출
    with torch.no_grad():
        embeddings = model(input_values).last_hidden_state.squeeze(0)  # (T, D)

    print(embeddings.shape)  # 예: (500, 768)