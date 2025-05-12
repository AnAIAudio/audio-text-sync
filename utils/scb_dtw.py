import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from audio.audio_util import change_audio_1st_dimension
from text.text_util import text_to_array


# 사코에-치바 대역 DTW 구현 (비용 행렬과 최적 경로 추적)
def sakoe_chiba_dtw(x, y, band_width):
    n = len(x)
    m = len(y)

    # 비용 행렬 초기화: 0으로 설정하지 않도록 최소값을 추가 (예: 1e-6)
    dtw_matrix = np.inf * np.ones((n, m))
    dtw_matrix[0, 0] = abs(x[0] - y[0]) + 1e-6  # 작은 값 추가

    # 동적 프로그래밍 계산 (사코에-치바 대역 적용)
    for i in range(1, n):
        for j in range(max(1, i - band_width), min(m, i + band_width + 1)):
            cost = abs(x[i] - y[j]) + 1e-6  # 작은 값 추가
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

    # 최적 경로 추적
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_cost = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
            if min_cost == dtw_matrix[i - 1, j]:
                i -= 1
            elif min_cost == dtw_matrix[i, j - 1]:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))

    path.reverse()
    return dtw_matrix, path


def run_dtw(audio_file_path, text_file_path: str):
    x = change_audio_1st_dimension(audio_file_path=audio_file_path)
    y = text_to_array(text_file_path=text_file_path)
    # x = np.array([1, 2, 3, 4, 5])
    # y = np.array([2, 2, 3, 4, 5])

    # 큰 배열을 방지하기 위한 다운 샘플링 예시
    max_len = 10000  # 필요에 따라 조절
    if len(x) > max_len:
        step_x = max(len(x) // max_len, 1)
        x = x[::step_x]
    if len(y) > max_len:
        step_y = max(len(y) // max_len, 1)
        y = y[::step_y]

    band_width = 1  # 대역 크기 설정

    # 두 시계열 간의 DTW 거리 계산 및 비용 행렬과 경로 얻기
    dtw_matrix, path = sakoe_chiba_dtw(y, x, band_width)

    # 6. DTW 비용 행렬 시각화 (히트맵)
    plt.figure(figsize=(10, 6))
    sns.heatmap(dtw_matrix, cmap='YlGnBu', cbar=True)
    plt.title("DTW Distance Matrix with Sakoe-Chiba Band")
    plt.xlabel("Text Data (y)")
    plt.ylabel("Audio Data (x)")

    # 7. 최적 경로 시각화
    for (i, j) in path:
        plt.plot(j, i, marker="o", color="red", markersize=3)

    plt.show()


def audio_text_mapping():
    import numpy as np
    import librosa
    from scipy.spatial.distance import cdist
    import json
    import matplotlib.pyplot as plt

    # 1. 오디오 파일에서 MFCC 추출
    def extract_mfcc(audio_path):
        # 오디오 로드
        y, sr = librosa.load(audio_path)
        # MFCC 특징 추출 (13차원)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc.T  # Time x Features 형태로 반환

    # 2. 텍스트 파일 준비 (문장별로 나누어 타임스탬프를 매핑할 준비)
    def prepare_text_timestamps(text):
        """
        텍스트를 문장별로 나누고, 각 문장의 시작과 끝 시간을 설정 (예시)
        예시 텍스트: "Hello. How are you? I am fine."
        """
        sentences = text.split('.')
        timestamps = []
        start_time = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                end_time = start_time + len(sentence) * 0.1  # 간단한 예시: 문자 수에 비례하여 시간 배정
                timestamps.append({"sentence": sentence, "start_time": start_time, "end_time": end_time})
                start_time = end_time
        return timestamps

    # 3. DTW를 통한 텍스트와 오디오의 동기화
    def dtw_distance(mfcc_audio, words_timestamps):
        """
        Dynamic Time Warping (DTW) Distance 계산
        - mfcc_audio: 오디오에서 추출한 MFCC (Time x Features)
        - words_timestamps: 텍스트 단어의 타임스탬프 (단어 x 3, (start_time, end_time, sentence))
        """
        num_frames = mfcc_audio.shape[0]
        num_words = len(words_timestamps)

        # DTW 거리 행렬 초기화
        dtw_matrix = np.zeros((num_frames, num_words))

        # 거리 계산 (오디오 프레임과 단어 간의 유사도 계산)
        for i in range(num_frames):
            for j in range(num_words):
                start_time, end_time, _ = words_timestamps[j]["start_time"], words_timestamps[j]["end_time"], \
                    words_timestamps[j]["sentence"]
                word_mfcc = mfcc_audio[i]  # 오디오 프레임의 MFCC 벡터
                # 단어의 시간 범위에 해당하는 평균 MFCC를 계산 (여기서는 단순화)
                word_mfcc_mean = np.mean(mfcc_audio[int(start_time):int(end_time)], axis=0) if int(
                    end_time) < num_frames else np.mean(mfcc_audio[int(start_time):], axis=0)
                dist = np.linalg.norm(word_mfcc - word_mfcc_mean)
                dtw_matrix[i, j] = dist

        # DTW 거리 행렬 시각화
        plt.imshow(dtw_matrix, cmap='hot', interpolation='nearest')
        plt.title("DTW Distance Matrix")
        plt.xlabel("Words")
        plt.ylabel("Audio Frames")
        plt.colorbar()
        plt.show()

        # Dynamic Programming을 통해 최단 거리 계산
        for i in range(1, num_frames):
            dtw_matrix[i, 0] += dtw_matrix[i - 1, 0]
        for j in range(1, num_words):
            dtw_matrix[0, j] += dtw_matrix[0, j - 1]

        for i in range(1, num_frames):
            for j in range(1, num_words):
                min_cost = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
                dtw_matrix[i, j] += min_cost

        # 동기화된 타임스탬프 생성
        result = []
        for j in range(num_words):
            sentence = words_timestamps[j]["sentence"]
            start_time = words_timestamps[j]["start_time"]
            end_time = words_timestamps[j]["end_time"]
            result.append({"sentence": sentence, "start_time": start_time, "end_time": end_time})

        # JSON 형식으로 출력
        return json.dumps(result, indent=4)

    # 4. 예시 데이터 (오디오 파일 경로와 텍스트 준비)
    audio_path = 'your_audio_file.wav'
    text = "Hello. How are you? I am fine."

    # MFCC 추출
    mfcc_audio = extract_mfcc(audio_path)

    # 텍스트와 타임스탬프 준비
    words_timestamps = prepare_text_timestamps(text)

    # DTW 거리 계산 후 동기화된 타임스탬프 JSON 생성
    synchronized_json = dtw_distance(mfcc_audio, words_timestamps)
    print(synchronized_json)
