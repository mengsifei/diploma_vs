import numpy as np
import torch
import re
from datasets.augmentation_methods import *

def score_essay(topic, essay, tokenizer, model, device):
    combined_text = f"[TOPIC] {topic} [TOPIC] {topic} [ESSAY] {essay}"
    inputs = tokenizer.encode_plus(
        combined_text,
        None,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_token_type_ids=False,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        outputs = outputs.cpu().numpy()[0]
        outputs = np.round(outputs * 2) / 2
    outputs = np.clip(outputs, 1, 9)
    return outputs

def test(best_model, tokenizer, test_df, device, prompt_id=1, essay_id=100):
    rubrics = ['Task Response', 'Coherence and Cohesion',
       'Lexical Resource', 'Grammatical Range and Accuracy']
    topic = test_df.iloc[prompt_id].prompt
    right_essay = test_df.iloc[prompt_id].essay
    swapped_essay = test_df.iloc[essay_id].essay
    print("++++++++++++++++++++++++++++")
    print("CASE_0: Wihtout edition")
    print("RUBRIC SCORE FOR THE ESSAY")
    print(test_df.iloc[prompt_id][rubrics])
    print(score_essay(topic, right_essay, tokenizer, best_model, device))


    print("++++++++++++++++++++++++++++")
    print("CASE_1: Less than 20 words")
    edited_essay = ' '.join(right_essay.split()[:19])
    print(score_essay(topic, edited_essay, tokenizer, best_model, device))


    print("++++++++++++++++++++++++++++")
    print("CASE_2: Totally Off topic")
    print(score_essay(topic, swapped_essay, tokenizer, best_model, device))


    print("++++++++++++++++++++++++++++")
    print("CASE_3: Partially Off topic")
    partially_essay = test_df.iloc[20].essay
    print("RUBRIC SCORE FOR THE NEW ESSAY")
    print(test_df.iloc[20][rubrics])
    print("Old prompt:", topic)
    print("New prompt:", test_df.iloc[20].prompt)
    print(score_essay(topic, partially_essay, tokenizer, best_model, device))


    print("++++++++++++++++++++++++++++")
    print("CASE_4: Spelling mistakes")
    edited_essay = misspell_text(right_essay)
    print(score_essay(topic, edited_essay, tokenizer, best_model, device))


    print("++++++++++++++++++++++++++++")
    print("CASE_5: No punctuations")
    punctuation_pattern = r'[^\w\s]'
    print(score_essay(topic, re.sub(punctuation_pattern, '', right_essay), tokenizer, best_model, device))
    

    print("++++++++++++++++++++++++++++")
    print("CASE_6: No paragraphing")
    print(score_essay(topic, right_essay.replace('\n', ' '), tokenizer, best_model, device))


    print("++++++++++++++++++++++++++++")
    print("CASE_7: Totally random text")
    random_text = generate_random_text()
    print(score_essay(topic, random_text, tokenizer, best_model, device))
    
    