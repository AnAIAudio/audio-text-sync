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


def visualize(audio_path, correct_srt_path, created_srt_path):
    import librosa, librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    wav, sr = librosa.load(audio_path, sr=16000)
    duration = len(wav) / sr

    # ===== ⓵ 예측 자막(TSV나 SRT 파싱) =====
    pred_segments = read_srt(created_srt_path)  # 예측 자막

    # ===== ⓶ (선택) 정답 자막 =====
    gt_segments = read_srt(correct_srt_path)  # (선택) 정답 자막

    # ===== ⓷ 스펙트럼 계산 =====
    S = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=1024, hop_length=320, n_mels=64
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    times = np.arange(S_dB.shape[1]) * (320 / sr)  # hop_length / sr

    fig, ax = plt.subplots(figsize=(12, 7))

    # ─── 1) 스펙트럼 ───
    img = librosa.display.specshow(
        S_dB, x_coords=times, y_axis="mel", sr=sr, hop_length=320, ax=ax, cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel-Spectrogram with Pred / GT Segments")

    # ─── 2) 예측 바 ───
    for start, end, txt in pred_segments:
        # ax.add_patch(
        #     plt.Rectangle(
        #         (start, 0),
        #         end - start,
        #         S_dB.shape[0],
        #         facecolor="none",
        #         edgecolor="cyan",
        #         linewidth=2,
        #         label="Predicted" if txt == pred_segments[0][2] else "",
        #     )
        # )

        mel_bins = S_dB.shape[0]  # y축이 mel-bin index
        ax.add_patch(
            plt.Rectangle(
                (start, 0),
                end - start,
                mel_bins,
                facecolor="none",
                edgecolor="cyan",
                linewidth=2,
            )
        )
        ax.text(start, S_dB.shape[0] + 2, txt, color="cyan", fontsize=9)

    # ─── 3) (옵션) 정답 바 ───
    for start, end, txt in gt_segments:
        # ax.add_patch(
        #     plt.Rectangle(
        #         (start, 0),
        #         end - start,
        #         S_dB.shape[0],
        #         facecolor="none",
        #         edgecolor="lime",
        #         linestyle="--",
        #         linewidth=2,
        #         label="Ground Truth" if txt == gt_segments[0][2] else "",
        #     )
        # )

        mel_bins = S_dB.shape[0]  # y축이 mel-bin index
        ax.add_patch(
            plt.Rectangle(
                (start, 0),
                end - start,
                mel_bins,
                facecolor="none",
                edgecolor="cyan",
                linewidth=2,
            )
        )
        ax.text(start, S_dB.shape[0] + 2, txt, color="cyan", fontsize=9)

    # ─── 4) DTW Cost Matrix 하단 인셋 ───
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cost_inset = inset_axes(ax, width="35%", height="35%", loc="upper right")

    # alignment.costMatrix, alignment.index1, index2 는 앞 단계 DTW 결과 그대로 사용
    # 예시용 무작위 행렬
    dummy = np.random.rand(30, 200)
    cost_inset.imshow(dummy, origin="lower", cmap="hot", aspect="auto")
    cost_inset.plot(np.arange(30), np.linspace(0, 199, 30), color="cyan", linewidth=1)
    cost_inset.set_title("DTW Cost + Path")
    cost_inset.set_xlabel("Text Idx")
    cost_inset.set_ylabel("Frame Idx")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left")
    ax.set_xlabel("Time (sec)")
    ax.set_xlim(0, duration)
    plt.tight_layout()
    plt.savefig("alignment_visual.png", dpi=150)
    plt.show()
