import numpy as np
import torch
import re
from datasets.augmentation_methods import *


def score_essay_vanilla(topic, essay, tokenizer, model_dict, device):
    essay = essay.replace("\n", "[SEP]")
    combined_text = f"[TOPIC] {topic} [ESSAY] {essay}"
    inputs = tokenizer.encode_plus(    
        combined_text,
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
    output_scores = []
    
    with torch.no_grad():
        # Evaluating each model on its tasks
        for key, model in model_dict.items():
            outputs = model(**inputs)
            outputs = outputs.cpu().numpy()[0]
            outputs = np.round(outputs)
            output_scores.extend(outputs)  # Assuming ordering in model_dict matches the task ordering
    output_scores = np.clip(output_scores, 1, 9)
    return output_scores

def test(model1, model2, tokenizer, test_df, device, prompt_id=1, essay_id=100):
    rubrics = ['Task Response', 'Coherence and Cohesion', 'Lexical Resource', 'Grammatical Range and Accuracy']
    score_essay = score_essay_vanilla
    topic = test_df.iloc[prompt_id].prompt
    right_essay = test_df.iloc[prompt_id].essay
    swapped_essay = test_df.iloc[essay_id].essay

    # Model dictionary according to the specific tasks they handle
    model_dict = {'model1': model1, 'model2': model2}

    print("++++++++++++++++++++++++++++")
    print("CASE_0: Without edition")
    print("RUBRIC SCORE FOR THE ESSAY")
    print(test_df.iloc[prompt_id][rubrics])
    print(score_essay(topic, right_essay, tokenizer, model_dict, device))

    print("++++++++++++++++++++++++++++")
    print("CASE_1: Less than 20 words")
    edited_essay = ' '.join(right_essay.split()[:19])
    print(score_essay(topic, edited_essay, tokenizer, model_dict, device))

    print("++++++++++++++++++++++++++++")
    print("CASE_2: Totally Off topic")
    print("Prompt:", topic)
    print("New prompt:", test_df.iloc[essay_id].prompt)
    print("Swapped essay's rubrics scores")
    print(test_df.iloc[essay_id][rubrics])
    print(score_essay(topic, swapped_essay, tokenizer, model_dict, device))

    print("++++++++++++++++++++++++++++")
    print("CASE_4: Spelling mistakes")
    print("Proportion = 0.9")
    edited_essay = misspell_text(right_essay, 0.9)
    print(score_essay(topic, edited_essay, tokenizer, model_dict, device))
    print("Proportion = 0.5")
    edited_essay = misspell_text(right_essay, 0.5)
    print(score_essay(topic, edited_essay, tokenizer, model_dict, device))
    print("Proportion = 0.2")
    edited_essay = misspell_text(right_essay, 0.2)
    print(score_essay(topic, edited_essay, tokenizer, model_dict, device))

    print("++++++++++++++++++++++++++++")
    print("CASE_5: No punctuations")
    punctuation_pattern = r'[^\w\s]'
    print(score_essay(topic, re.sub(punctuation_pattern, '', right_essay), tokenizer, model_dict, device))

    print("++++++++++++++++++++++++++++")
    print("CASE_6: No paragraphing")
    print(score_essay(topic, right_essay.replace('\n', ' '), tokenizer, model_dict, device))

    print("++++++++++++++++++++++++++++")
    print("CASE_7: Totally random text")
    random_text = generate_random_text()
    print(score_essay(topic, random_text, tokenizer, model_dict, device))
