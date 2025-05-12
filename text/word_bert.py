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
    inputs = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
    offsets = inputs.pop("offset_mapping")[0].numpy()  # shape: (tokens, 2)
    input_ids = inputs["input_ids"][0]
    tokens_all = tokenizer.convert_ids_to_tokens(input_ids)
    # 4. [CLS], [SEP] 제외
    valid_idxs = list(range(1, len(tokens_all) - 1))  # skip [CLS], [SEP]
    tokens = [tokens_all[i] for i in valid_idxs]
    offsets = [offsets[i] for i in valid_idxs]

    # 5. 토큰이 어느 문장에 속하는지 판단
    token_sentence_ids = []
    last_valid_id = None
    for start_offset, end_offset in offsets:
        matched = False
        for sent_id, (s_start, s_end) in enumerate(sentence_spans):
            if s_start <= start_offset < s_end:
                token_sentence_ids.append(sent_id)
                last_valid_id = sent_id
                matched = True
                break
        if not matched:
            # 유효한 offset이 없는 토큰: 직전 문장 ID를 복사 (e.g. subword)
            token_sentence_ids.append(last_valid_id if last_valid_id is not None else 0)

    with torch.no_grad():
        outputs = bert_model(**inputs)
    token_embeddings = outputs.last_hidden_state[0][1:-1].numpy()  # exclude [CLS] and [SEP]
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
