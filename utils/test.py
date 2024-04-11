import numpy as np
import torch
import random
import re

def misspell_word(word):
    action = random.choice(['delete', 'repeat']) 
    letters = list(word)
    if len(letters) > 1:
        if action == 'delete':
            del letters[random.choice(range(len(letters)))]
        elif action == 'repeat':
            repeat_index = random.choice(range(len(letters)))
            letters.insert(repeat_index, letters[repeat_index])
    return ''.join(letters)

def misspell_text(text, proportion=0.5):
    words = text.split()
    num_to_misspell = int(len(words) * proportion)
    indices_to_misspell = random.sample(range(len(words)), num_to_misspell)
    for i in indices_to_misspell:
        words[i] = misspell_word(words[i])
    return ' '.join(words)

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
    print("Prompt 1 and Essay 100")
    print("===========")
    topic = test_df.iloc[prompt_id].prompt
    essay = test_df.iloc[essay_id].essay
    print("++++++++++++++++++++++++++++")
    print("CASE_0: Wihtout edition")
    print("RUBRIC SCORE FOR THE ESSAY")
    print(test_df.iloc[essay_id][rubrics])
    print(score_essay(test_df.iloc[prompt_id][rubrics], essay, tokenizer, best_model, device))
    print("++++++++++++++++++++++++++++")
    print("CASE_1: Less than 20 words")
    
    edited_essay = ' '.join(essay.split()[:19])
    print(score_essay(topic, edited_essay, tokenizer, best_model, device))


    print("++++++++++++++++++++++++++++")
    print("CASE_2: Totally Off topic")
    print(score_essay(topic, essay, tokenizer, best_model, device))


    print("++++++++++++++++++++++++++++")
    print("CASE_3: Partially Off topic")
    rubric_ranges = [[], [], [], []]
    essay = test_df.iloc[20].essay
    print("RUBRIC SCORE FOR THE NEW ESSAY")
    print(test_df.iloc[20][rubrics])
    print(score_essay(topic, essay, tokenizer, best_model, device))


    print("++++++++++++++++++++++++++++")
    print("CASE_4: Spelling mistakes")
    rubric_ranges = [[], [], [], []]
    essay = test_df.iloc[essay_id].essay
    edited_essay = misspell_text(essay)
    print(score_essay(topic, edited_essay, tokenizer, best_model, device))


    print("++++++++++++++++++++++++++++")
    print("CASE_5: No punctuations")
    punctuation_pattern = r'[^\w\s]'
    print(score_essay(topic, re.sub(punctuation_pattern, '', essay), tokenizer, best_model, device))
    

    print("++++++++++++++++++++++++++++")
    print("CASE_6: No paragraphing")
    print(score_essay(topic, essay.replace('\n', ' '), tokenizer, best_model, device))
    
    