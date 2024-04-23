import torch
from torch.utils.data import Dataset
import re
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.df = df
        self.text = df['essay']
        self.prompt = df['prompt']
        self.max_len = max_len
        self.labels = df[['Task Response', 'Coherence and Cohesion', 'Lexical Resource', 'Grammatical Range and Accuracy']].values
    def __len__(self):
        return len(self.text)
    def calculate_features(self, text):
        paragraph_pattern = re.compile(r'\n')
        num_paragraphs = len(paragraph_pattern.findall(text)) + 1
        features = np.array([num_paragraphs], dtype=np.float32)
        return 1 / features
    def __getitem__(self, index):
        text = self.text[index].replace('\n', '[SEP]')
        prompt = self.prompt[index]
        combined_text = f"[prompt] {prompt} [prompt] {prompt} [ESSAY] {text}"
        inputs = self.tokenizer.encode_plus(
            combined_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs['token_type_ids'].flatten(),
            'labels': torch.FloatTensor(self.labels[index])
        }           
