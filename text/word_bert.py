def run_word_bert(sentences: list[str], full_text: str):
    import torch
    from sklearn.preprocessing import normalize
    from transformers import BertTokenizerFast, BertModel

    # 2. 문장별 텍스트 범위 추적
    sentence_spans = []
    start = 0
    for s in sentences:
        end = start + len(s)
        sentence_spans.append((start, end))
        start = end + 1  # 띄어쓰기 포함

    # 3. BERT 모델 및 토크나이저
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = inputs.pop("offset_mapping")[0].numpy()  # (num_tokens, 2)

    # 4. 각 토큰이 어떤 문장에 속하는지 판별
    token_sentence_ids = []
    for start_offset, end_offset in offsets[1:-1]:  # skip [CLS], [SEP]
        token_start = start_offset
        for sent_id, (span_start, span_end) in enumerate(sentence_spans):
            if span_start <= token_start < span_end:
                token_sentence_ids.append(sent_id)
                break

    # 5. BERT 문맥 기반 단어 임베딩
    with torch.no_grad():
        outputs = bert_model(**inputs)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[
        1:-1
    ]  # [CLS], [SEP] 제외
    token_embeddings = outputs.last_hidden_state[0][
        1:-1
    ].numpy()  # shape: (num_tokens, 768)
    token_embeddings = normalize(token_embeddings)

    return tokens, token_embeddings, token_sentence_ids


def run_visualize(alignment):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.imshow(alignment.costMatrix.T, origin="lower", cmap="hot", aspect="auto")
    plt.plot(alignment.index1, alignment.index2, color="cyan", linewidth=1)
    plt.title("DTW Alignment Path (Text Tokens ↔ Audio Frames)")
    plt.xlabel("Text Token Index")
    plt.ylabel("Audio Frame Index")
    plt.colorbar(label="Distance")
    plt.tight_layout()
    plt.show()
