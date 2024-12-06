from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_TOKENS = 1024  # BART context size

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_summary(input_ids, max_length: int = 150, min_length: int = 30) -> str:
    summary_ids = model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def chunk_text(text: str, max_input_tokens: int = MAX_INPUT_TOKENS) -> List[List[int]]:
    tokens = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
    input_ids = tokens["input_ids"][0]

    chunks = []
    for i in range(0, len(input_ids), max_input_tokens):
        chunk_ids = input_ids[i:i+max_input_tokens]
        chunks.append(chunk_ids)
    return chunks

def summarize_large_text(
    text: str,
    max_length: int = 150,
    min_length: int = 30,
    max_input_tokens: int = MAX_INPUT_TOKENS
) -> List[str]:
    # If it fits in one chunk:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    if tokens["input_ids"].shape[1] <= max_input_tokens:
        return [generate_summary(tokens["input_ids"], max_length, min_length)]

    # Otherwise, chunk it
    chunk_ids_list = chunk_text(text, max_input_tokens=max_input_tokens)
    partial_summaries = []
    for chunk_ids in chunk_ids_list:
        summary = generate_summary(chunk_ids.unsqueeze(0), max_length, min_length)
        partial_summaries.append(summary)

    return partial_summaries
