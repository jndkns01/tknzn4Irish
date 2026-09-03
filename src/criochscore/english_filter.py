"""Optional English-word filtering for corpus cleaning, using Facebook's
fastText language-identification model. Only imported when
``--filter-english`` is passed, so users who don't need it don't need the
``fasttext``/``huggingface_hub`` dependencies installed.
"""

from __future__ import annotations


def find_english_words(tokens: list[str], k: int = 5) -> list[str]:
    """Return the subset of ``tokens`` fastText identifies as English (and
    not also plausibly Irish)."""
    import fasttext
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification", filename="model.bin"
    )
    model = fasttext.load_model(model_path)

    english_words = []
    for word in tokens:
        labels = model.predict(word.lower(), k=k)[0]
        if "__label__eng_Latn" in labels and "__label__gle_Latn" not in labels:
            english_words.append(word)
    return english_words
