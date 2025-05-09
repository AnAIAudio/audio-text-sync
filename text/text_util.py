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
