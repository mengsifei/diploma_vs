import numpy as np
import torch
import re
from my_datasets.augmentation_methods import *

def score_essay_vanilla(topic, essay, tokenizer, model, device):
    essay = essay.replace("\n", "[SEP]")
    combined_text = f"[TOPIC] {topic} [TOPIC] {topic} [ESSAY] {essay}"
    inputs = tokenizer.encode_plus(    
        combined_text,
        None,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
        return_token_type_ids=True 
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.cpu().numpy()[0]
        scores = np.round(logits).astype(int)
    scores = np.clip(scores, 1, 9)
    return scores

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
        logits = outputs.cpu().numpy()[0]
        scores = np.round(logits).astype(int) 
    scores = np.clip(scores, 1, 9)
    return scores

def test(model, tokenizer, test_df, device, prompt_id=1, essay_id=100):
    rubrics = ['Task Response', 'Coherence and Cohesion', 'Lexical Resource', 'Grammatical Range and Accuracy']
    topic = test_df.iloc[prompt_id].prompt
    right_essay = test_df.iloc[prompt_id].essay
    swapped_essay = test_df.iloc[essay_id].essay
    print("++++++++++++++++++++++++++++")
    print("CASE_0: Without edition")
    print("RUBRIC SCORE FOR THE ESSAY")
    print(test_df.iloc[prompt_id][rubrics])
    print(score_essay_hier(topic, right_essay, tokenizer, model, device))
    
    print("++++++++++++++++++++++++++++")
    print("CASE_1: Less than 20 words")
    edited_essay = ' '.join(right_essay.split()[:19])
    print(score_essay_hier(topic, edited_essay, tokenizer, model, device))

    print("++++++++++++++++++++++++++++")
    print("CASE_2: Totally Off topic")
    print("Prompt:", topic)
    print("New prompt:", test_df.iloc[essay_id].prompt)
    print("Swapped essay's rubrics scores")
    print(test_df.iloc[essay_id][rubrics])
    print(score_essay_hier(topic, swapped_essay, tokenizer, model, device))

    print("++++++++++++++++++++++++++++")
    print("CASE_4: Spelling mistakes")
    print("Proportion = 0.9")
    edited_essay = misspell_text(right_essay, 0.9)
    print(score_essay_hier(topic, edited_essay, tokenizer, model, device))
    print("Proportion = 0.5")
    edited_essay = misspell_text(right_essay, 0.5)
    print(score_essay_hier(topic, edited_essay, tokenizer, model, device))
    print("Proportion = 0.2")
    edited_essay = misspell_text(right_essay, 0.2)
    print(score_essay_hier(topic, edited_essay, tokenizer, model, device))

    print("++++++++++++++++++++++++++++")
    print("CASE_5: No punctuations")
    punctuation_pattern = r'[^\w\s]'
    print(score_essay_hier(topic, re.sub(punctuation_pattern, '', right_essay), tokenizer, model, device))

    print("++++++++++++++++++++++++++++")
    print("CASE_6: No paragraphing")
    print(score_essay_hier(topic, right_essay.replace('\n', ' '), tokenizer, model, device))
    