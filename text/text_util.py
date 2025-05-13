from typing import TypeVar, List, Sequence, Optional


def text_to_array(text_file_path: str):
    import os

    if not os.path.exists(text_file_path):
        raise FileNotFoundError(f"Text file not found: {text_file_path}")

    from sklearn.feature_extraction.text import CountVectorizer

    text = "I love machine learning"
    with open(text_file_path, "r") as f:
        text = f.read()

    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform([text])
    return x.toarray().flatten()


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


from itertools import cycle, islice

T = TypeVar("T")


def take_n(
    items: Sequence[T],
    n: int,
    *,
    on_short: str = "truncate",  # "truncate"│"strict"│"pad"│"cycle"
    pad_value: Optional[T] = None,  # on_short="pad" 일 때 채울 값
) -> List[T]:
    """
    items     : 원본 시퀀스 (list·tuple 등 인덱서블 자료형)
    n         : 가져올 개수 (음수면 ValueError)
    on_short  : items 길이 < n 일 때 행동
        ─ "truncate" : 있는 만큼만 반환           (기본값)  ex) len=3, n=5 → 3개 반환
        ─ "strict"   : ValueError 발생
        ─ "pad"      : 부족분을 pad_value로 채워 길이를 n으로 맞춰 반환
        ─ "cycle"    : 리스트를 반복(cycle)해서 n개 채워 반환
    pad_value : on_short="pad" 에서 쓰는 채움 값
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    length = len(items)

    if n <= length:
        # 정상 범위: 슬라이싱만 하면 됨
        return list(items[:n])

    # --- 리스트가 짧은 경우 ---
    if on_short == "truncate":
        return list(items)  # 있는 만큼만
    elif on_short == "strict":
        raise ValueError(
            f"Requested {n} items but only {length} available (on_short='strict')."
        )
    elif on_short == "pad":
        # pad_value가 None이면 에러가 더 직관적이라 그대로 둠
        padded = list(items) + [pad_value] * (n - length)
        return padded
    elif on_short == "cycle":
        return list(islice(cycle(items), n))
    else:
        raise ValueError(
            f"Invalid on_short mode: {on_short!r}. "
            "Choose from 'truncate', 'strict', 'pad', 'cycle'."
        )


class SequentialPicker:
    def __init__(self, items: Sequence[T]):
        self.items = list(items)
        self.idx = 0

    def take(self, n: int, *, on_short: str = "truncate", pad_value=None) -> List[T]:
        start, end = self.idx, self.idx + n
        self.idx = min(end, len(self.items))  # 포인터 이동

        if end <= len(self.items):
            return self.items[start:end]

        # 남은 게 부족할 때
        rest = self.items[start:]
        shortage = n - len(rest)
        if on_short == "truncate":
            return rest
        elif on_short == "pad":
            return rest + [pad_value] * shortage
        elif on_short == "cycle":
            return rest + list(islice(cycle(self.items), shortage))
        else:  # "strict"
            raise ValueError("Not enough items left")


import re
from typing import List, Dict, Callable

Segment = Dict[str, float | str]  # 최소한 start, end, text 만 있다고 가정

# ──────────────────────────────────────────────────────────────
# 1) “문장 완결” 판정 함수
#    Whisper는 대부분 마침표/물음표/느낌표를 넣어 주므로
#    텍스트가 그 기호로 끝나면 ‘완전한 문장’으로 보겠다는
#    가장 단순한 정의입니다. 필요하면 더 정교하게 바꾸세요.
# ──────────────────────────────────────────────────────────────
_END_RE = re.compile(r'[.!?…]\s*["»”’)]*\s*$')


def is_complete(text: str) -> bool:
    return bool(_END_RE.search(text))


# ──────────────────────────────────────────────────────────────
# 2) 머지 함수
# ──────────────────────────────────────────────────────────────
def merge_segments(
    segments: List[Segment],
    picker: SequentialPicker,
    complete_fn: Callable[[str], bool] = is_complete,
) -> List[Segment]:
    """
    segments   : Whisper segment dict들의 리스트
    complete_fn: buffer가 '완전한 문장'인지 판단하는 함수
    반환        : 병합된 segment 리스트
    """
    merged: List[Segment] = []
    buf_text = ""
    buf_start = None

    for seg in segments:
        # 버퍼가 비어 있으면 새 문장 시작
        if not buf_text:
            buf_start = seg["start"]

        # ① 이어 붙이고, ② 문장 완결 검사
        buf_text = f"{buf_text} {seg['text']}".strip()

        if not complete_fn(buf_text):
            continue

        dddd = split_sentences(text=buf_text)
        zzzz = len(dddd)

        picker_list = picker.take(n=zzzz)
        picker_text = "".join(picker_list)

        merged.append(
            {
                "start": buf_start,
                "end": seg["end"],
                "text": picker_text,
            }
        )
        buf_text = ""  # 버퍼 초기화
        buf_start = None  # 시작 시간 리셋

    # 루프 종료 후에도 남은 텍스트가 있으면 마지막 문장으로 취급
    if buf_text:
        last_end = segments[-1]["end"] if segments else 0.0
        merged.append({"start": buf_start or 0.0, "end": last_end, "text": buf_text})

    return merged
