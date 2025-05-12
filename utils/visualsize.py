def read_srt(path):
    """SRT 파일을 읽어 [(start_sec, end_sec, text), ...] 리스트 반환"""
    from pathlib import Path
    import srt

    srt_text = Path(path).read_text(encoding="utf-8")
    subs = list(srt.parse(srt_text))

    import datetime

    def to_sec(td: datetime.timedelta):
        return td.total_seconds()

    segments = [
        (to_sec(sub.start), to_sec(sub.end), sub.content.strip()) for sub in subs
    ]
    return segments


def visualize(alignment, audio_path, correct_srt_path, created_srt_path):
    # 설치: pip install librosa matplotlib pandas srt dtw-python transformers torchaudio scikit-learn
    import librosa, librosa.display, matplotlib.pyplot as plt
    import numpy as np, pandas as pd, srt, datetime
    from pathlib import Path

    # ────────────────────────────── 0. 입 력 ──────────────────────────────
    wav_path = audio_path  # 16 kHz mono
    pred_srt = created_srt_path  # 예측
    gt_srt = correct_srt_path  # (선택) 정답 SRT

    # DTW 결과 (alignment) 는 앞 단계에서 이미 계산했다고 가정
    # alignment.costMatrix, alignment.index1, alignment.index2, text_len, audio_len  제공
    # ─────────────────────────────────────────────────────────────────────

    # 1) wav 로드 및 Mel-스펙트럼
    wav, sr = librosa.load(wav_path, sr=16000, mono=True)
    S = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=1024, hop_length=320, n_mels=64
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    times = np.arange(S_dB.shape[1]) * (320 / sr)  # hop_length / sr

    # 2) stride - 실측 기반 재계산 ①
    duration = len(wav) / sr
    hidden_len = (
        alignment.costMatrix.shape[1] - 1
    )  # padding 열 제외 (=오디오 프레임 수)
    stride = duration / hidden_len

    pred_segments = read_srt(pred_srt)
    gt_segments = read_srt(gt_srt)  # 주석 처리하면 정답 없이 작동

    # ─────────────────────────── 4. 시 각 화 ────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))

    # 스펙트럼
    ax.set_title("Mel-Spectrogram with Pred / GT Segments")
    img = librosa.display.specshow(
        S_dB,
        x_coords=times,
        y_axis="mel",
        sr=sr,
        hop_length=320,
        cmap="magma",
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # 0. 기본: wav, sr, S_dB, times 계산은 동일
    # mel 축은 위가 0 Hz, 아래가 8 kHz 이므로 보통 y0 > y1
    y0, y1 = ax.get_ylim()
    y_lower, y_upper = (min(y0, y1), max(y0, y1))  # 항상 작은→큰 순으로

    # ── ① y-축 전체 덮도록 Rectangle
    for i, (st, ed, txt) in enumerate(pred_segments):
        ax.add_patch(
            plt.Rectangle(
                (st, y_lower),  # (x, y)
                ed - st,  # width
                y_upper - y_lower,  # height  ← 축 전체
                facecolor="none",
                edgecolor="cyan",
                linewidth=2,
                label="Predicted" if i == 0 else "",
            )
        )
        ax.text(st, y_upper + 200, txt, rotation=90, color="cyan", fontsize=8)

    for i, (st, ed, txt) in enumerate(gt_segments):
        ax.add_patch(
            plt.Rectangle(
                (st, y_lower),
                ed - st,
                y_upper - y_lower,
                facecolor="none",
                edgecolor="lime",
                linestyle="--",
                linewidth=2,
                label="Ground Truth" if i == 0 else "",
            )
        )

    ax.set_xlim(0, duration)
    ax.set_xlabel("Time (sec)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left")

    # ── DTW cost-matrix 인셋 ③
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # ── ③ DTW 히트맵 인셋 보정
    cm = alignment.costMatrix[1:, 1:]
    path_x = alignment.index1 - 1
    path_y = alignment.index2 - 1
    # cm = alignment.costMatrix[1:, 1:]  # padding 제거
    # path_x = alignment.index1 - 1  # 패딩 보정
    # path_y = alignment.index2 - 1

    cost_ax = inset_axes(ax, width="35%", height="35%", loc="upper right")
    cost_ax.imshow(cm.T, origin="lower", cmap="hot", aspect="auto")
    cost_ax.plot(path_x, path_y, color="cyan", lw=1)
    cost_ax.set_title("DTW Cost + Path")
    cost_ax.set_xlabel("Text Idx")
    cost_ax.set_ylabel("Frame Idx")

    plt.tight_layout()
    plt.savefig("alignment_visual_fixed.png", dpi=150)
    plt.show()
