from typing import TypeVar, List, Sequence, Optional


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
