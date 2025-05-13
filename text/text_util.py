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
