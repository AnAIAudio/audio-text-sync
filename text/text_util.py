from itertools import cycle, islice
from typing import TypeVar, Sequence
from typing import List, Dict


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


T = TypeVar("T")


class SequentialPicker:
    def __init__(self, items: Sequence[T]):
        self.items = list(items)
        self.idx = 0

    def move_back(self, steps=1):
        self.idx = max(0, self.idx - steps)

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


Segment = Dict[str, float | str]  # 최소한 start, end, text 만 있다고 가정


def is_complete(text: str) -> bool:
    import re

    _END_RE = re.compile(r'[.!?…]\s*["»”’)]*\s*$')

    return bool(_END_RE.search(text))


def merge_segments(segments: List[Segment], picker: SequentialPicker) -> List[Segment]:
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

        if not is_complete(buf_text):
            continue

        picker_list = picker.take(n=1)
        picker_text = "".join(picker_list)

        similar = calc_text_similarity(picker_text, buf_text)
        print("similar , text , buf_text : ", similar, picker_text, buf_text)
        if similar < 0.8:
            picker.move_back()
            continue

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


def calc_text_similarity(text1: str, text2: str) -> float:
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    emb_a, emb_b = model.encode([text1, text2], convert_to_tensor=True)
    sim = util.cos_sim(emb_a, emb_b)
    similarity = float(sim.item())
    return similarity
