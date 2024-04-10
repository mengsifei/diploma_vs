import torch
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.df = df
        self.text = df['essay']
        self.topic = df['prompt']
        self.labels = self.df[['Task Response', 'Coherence and Cohesion',
       'Lexical Resource', 'Grammatical Range and Accuracy']].values
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        topic = self.topic[index]
        combined_text = f"[TOPIC] {topic} [TOPIC] {topic} [ESSAY] {text}"
        inputs = self.tokenizer.encode_plus(
            combined_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.FloatTensor(list(self.labels[index]))
        }