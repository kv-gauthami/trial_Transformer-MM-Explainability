"""Locating the image-token span and target-word positions in LLaVA input_ids."""
import math


def image_token_span(input_ids, image_token_id):
    """Return (start, num_tokens, grid) of the contiguous image-token block.

    Requires a transformers version whose processor expands <image> into one
    placeholder per patch (576 for LLaVA-1.5-7B). Otherwise sequence positions
    in input_ids would not match positions in the attention matrices.
    """
    pos = [i for i, t in enumerate(input_ids) if t == image_token_id]
    if not pos:
        raise ValueError("No image tokens found in input_ids.")
    if len(pos) == 1:
        raise ValueError(
            "input_ids holds a single <image> token: this transformers version expands "
            "image tokens inside the model, so input positions are misaligned with the "
            "attention matrices. Upgrade transformers (processor-side expansion).")
    start, n = pos[0], len(pos)
    if pos[-1] - start + 1 != n:
        raise ValueError("Image tokens are not contiguous.")
    grid = int(math.isqrt(n))
    if grid * grid != n:
        raise ValueError(f"{n} image tokens do not form a square grid.")
    return start, n, grid


def _candidate_token_ids(tokenizer, word):
    """Token-id sequences the word may appear as (SentencePiece '▁word' variants)."""
    cands = []
    for text in (word, " " + word):
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = [i for i in ids if tokenizer.convert_ids_to_tokens(i) != "▁"]  # bare-space piece
        if ids and ids not in cands:
            cands.append(ids)
    return cands


def find_word_position(input_ids, tokenizer, word, search_from=0, occurrence="last"):
    """Find the position of a word in the input_ids.
    """
    matches = set()
    for cand in _candidate_token_ids(tokenizer, word):
        L = len(cand)
        for i in range(search_from, len(input_ids) - L + 1):
            if input_ids[i:i + L] == cand:
                matches.add(i)
    matches = sorted(matches)
    if not matches:
        tail = tokenizer.convert_ids_to_tokens(input_ids[search_from:])
        raise ValueError(f"'{word}' not found after position {search_from}. Tokens there: {tail}")
    if occurrence == "first":
        return matches[0]
    if occurrence == "last":
        return matches[-1]
    return matches[int(occurrence)]
