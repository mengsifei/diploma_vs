import random
import numpy as np
import torch
import re

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def process_chunks(tokens, tokenizer, max_len=512, max_chunks=3):
    chunk_size = max_len - 2
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)][:max_chunks]
    return encode_chunks(chunks, tokenizer, max_len, max_chunks)
def encode_chunks(chunks, tokenizer, max_len, max_chunks):
    input_ids, attention_masks, token_type_ids = [], [], []
    for chunk in chunks:
        if not chunk:
            chunk = ["[PAD]"]
        encoded = tokenizer.encode_plus(
            chunk,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            is_pretokenized=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'].squeeze(0))
        attention_masks.append(encoded['attention_mask'].squeeze(0))
        token_type_ids.append(encoded['token_type_ids'].squeeze(0))

    while len(input_ids) < max_chunks:
        input_ids.append(torch.zeros(max_len, dtype=torch.long))
        attention_masks.append(torch.zeros(max_len, dtype=torch.long))
        token_type_ids.append(torch.zeros(max_len, dtype=torch.long))
    return torch.stack(input_ids), torch.stack(attention_masks), torch.stack(token_type_ids)
def score_essay_hier(topic, essay, tokenizer, model, device):
    pattern = r'\n\w+'
    essay = re.sub(pattern, '\n', essay)
    essay = essay.replace("\n", "[SEP]")
    combined_text = f"[TOPIC] {topic} [TOPIC] {topic} [ESSAY] {essay}"
    tokenized_text = tokenizer.tokenize(combined_text)
    input_ids_doc, attention_mask_doc, token_type_ids_doc = process_chunks(tokenized_text, tokenizer, 512, 3)
    with torch.no_grad():
        input_ids_doc = input_ids_doc.unsqueeze(0).to(device)
        attention_mask_doc = attention_mask_doc.unsqueeze(0).to(device)
        token_type_ids_doc = token_type_ids_doc.unsqueeze(0).to(device)
        outputs = model(input_ids_doc, attention_mask_doc, token_type_ids_doc)
        logits = outputs.cpu().numpy()[0]  # assuming outputs to be of shape [1, num_labels]
        scores = np.round(logits).astype(int)  # rounding off to the nearest integer
    scores = np.clip(scores, 1, 9)  # ensuring scores are within the valid range
    return scores