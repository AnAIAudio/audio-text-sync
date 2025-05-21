from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np
from typing import List

from text.text_util import SequentialPicker, Segment

# 모델 및 토크나이저 로딩
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()


def score_sentence(sentence):
    """
    입력 문장에 있는 각 단어를 마스킹하고 그 단어의 점수를 계산해 전체 평균 점수를 반환합니다.
    점수가 낮을수록 자연스러운 문장입니다.
    """
    tokens = tokenizer.tokenize(sentence)
    input_ids = tokenizer.encode(sentence, return_tensors="pt")[0]

    loss_list = []
    for i in range(1, len(input_ids) - 1):  # [CLS] ~ [SEP] 제외
        masked_input = input_ids.clone()
        masked_input[i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked_input.unsqueeze(0))
            logits = outputs.logits

        softmax = torch.nn.functional.softmax(logits[0, i], dim=-1)
        prob = softmax[input_ids[i]]
        loss = -torch.log(prob + 1e-10)  # log likelihood

        loss_list.append(loss.item())

    avg_loss = np.mean(loss_list)
    return avg_loss


def is_sentence_natural(sentence, threshold=5.0):
    score = score_sentence(sentence)
    print(sentence)
    print(f"Score: {score:.4f}")
    return score < threshold


def bert_seperate(segments):
    merged_segments = []

    current = {"start": int(1e9), "end": -1, "text": ""}
    for segment in segments:
        current["text"] += f" {segment['text']}"
        current["start"] = min(current["start"], segment["start"])
        current["end"] = min(current["end"], segment["end"])

        if is_sentence_natural(current["text"]):
            merged_segments.append(current)
            current = {"start": int(1e9), "end": -1, "text": ""}
    else:
        if current["text"]:
            merged_segments.append(current)

    return merged_segments


if __name__ == "__main__":
    test_datas = [
        "welcome",
        "welcome to",
        "welcome to films",
        "welcome to films and",
        "welcome to films and stars",
        "welcome to films and stars your",
        "welcome to films and stars your all",
        "welcome to films and stars your all access",
        "welcome to films and stars your all access pass",
        "welcome to films and stars your all access pass to",
        "welcome to films and stars your all access pass to the",
        "welcome to films and stars your all access pass to the magic",
        "welcome to films and stars your all access pass to the magic mayhem",
        "welcome to films and stars your all access pass to the magic mayhem and",
        "welcome to films and stars your all access pass to the magic mayhem and megastars",
        "welcome to films and stars your all access pass to the magic mayhem and megastars of",
        "welcome to films and stars your all access pass to the magic mayhem and megastars of the",
        "welcome to films and stars your all access pass to the magic mayhem and megastars of the movie",
        "welcome to films and stars your all access pass to the magic mayhem and megastars of the movie world",
        "welcome to films and stars your all access pass to the magic mayhem and megastars of the movie world this",
        "welcome to films and stars your all access pass to the magic mayhem and megastars of the movie world this week",
    ]

    for data in test_datas:
        print(is_sentence_natural(data))
