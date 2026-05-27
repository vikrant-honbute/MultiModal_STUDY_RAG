def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    words = cleaned.split(" ")
    chunks: list[str] = []
    current_words: list[str] = []
    current_len = 0

    for word in words:
        extra = len(word) + (1 if current_words else 0)
        if current_words and current_len + extra > chunk_size:
            chunks.append(" ".join(current_words))

            # carry overlap from the end of the previous chunk
            if overlap > 0:
                carry_words: list[str] = []
                carry_len = 0
                for w in reversed(current_words):
                    w_len = len(w) + (1 if carry_words else 0)
                    if carry_len + w_len > overlap:
                        break
                    carry_words.insert(0, w)
                    carry_len += w_len
                current_words = carry_words
                current_len = carry_len
            else:
                current_words = []
                current_len = 0

        current_words.append(word)
        current_len += extra

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks
