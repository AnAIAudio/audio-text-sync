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

        nltk.download("punkt", quiet=True)
        from nltk.tokenize import sent_tokenize

        return [s.strip() for s in sent_tokenize(text)]


def create_text_line_full_text(raw_text: str, lang_code: str = "en"):

    sentences = []
    for line in raw_text.splitlines():
        if line.strip():  # 빈 줄 건너뛰기
            sentences.extend(split_sentences(line, lang_code))

    return "".join(sentences)
