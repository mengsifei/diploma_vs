import numpy as np
import torch
import re
from datasets.augmentation_methods import *

def score_essay_vanilla(topic, essay, tokenizer, model, device):
    inputs = tokenizer.encode_plus(
        [topic, essay.replace("\n", f" [SEP][SEP] ")],
        None,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_token_type_ids=True,
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


def score_essay_dual(topic, essay, tokenizer, model, device):
    essay_inputs = tokenizer(
        essay,
        max_length=512 - 60, 
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    essay_inputs = {'essay_input_ids': essay_inputs['input_ids'],
            'essay_attention_mask': essay_inputs['attention_mask']}

    topic_inputs = tokenizer(
        topic,
        max_length=60,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    topic_inputs = {'topic_input_ids': topic_inputs['input_ids'],
            'topic_attention_mask': topic_inputs['attention_mask']}
    essay_inputs = {k: v.to(device) for k, v in essay_inputs.items()}
    topic_inputs = {k: v.to(device) for k, v in topic_inputs.items()}
    
    with torch.no_grad():
        outputs = model(essay_input_ids=essay_inputs['essay_input_ids'], 
                            essay_attention_mask=essay_inputs['essay_attention_mask'], 
                            topic_input_ids=topic_inputs['topic_input_ids'], 
                            topic_attention_mask=topic_inputs['topic_attention_mask'])
        outputs = outputs.cpu().numpy()[0]
        outputs = np.round(outputs * 2) / 2
    outputs = np.clip(outputs, 1, 9)
    return outputs

def test(best_model, tokenizer, test_df, device, is_dual, prompt_id=1, essay_id=100):
    rubrics = ['Task Response', 'Coherence and Cohesion',
       'Lexical Resource', 'Grammatical Range and Accuracy']
    score_essay = score_essay_dual if is_dual else score_essay_vanilla
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
    
    