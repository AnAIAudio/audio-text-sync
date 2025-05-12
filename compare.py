def compare_dtw(text_embeds, audio_embeds):
    from dtw import dtw
    from scipy.spatial.distance import cosine

    # 3. 거리 계산 함수 정의 (cosine distance)
    def cosine_dist(x, y):
        return cosine(x, y)

    # 4. DTW 계산
    alignment = dtw(text_embeds, audio_embeds, dist=cosine_dist)

    return alignment
