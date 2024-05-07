import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.df = df
        self.text = df['essay']
        self.prompt = df['prompt']
        self.max_len = max_len
        self.max_chunks = 3
        self.labels = df[['Task Response', 'Coherence and Cohesion', 'Lexical Resource', 'Grammatical Range and Accuracy']].values
    def __len__(self):
        return len(self.text)
    def segment_text(self, text):
        paragraphs = text.strip().split('\n')
        paragraphs = [p.replace('\n', ' [SEP] ') for p in paragraphs]  # Replace newlines within paragraphs if any
        if len(paragraphs) < 3:
            return paragraphs + [''] * (3 - len(paragraphs))  # Ensure there are always 3 chunks
        intro = paragraphs[0] + ' [SEP] '
        body = ' [SEP] '.join(paragraphs[1:-1]) + ' [SEP] '  # Join middle paragraphs with SEP
        conclusion = paragraphs[-1] + ' [SEP] '
        return [intro, body, conclusion]
    def process_chunks(self, chunks):
        input_ids, attention_masks, token_type_ids = [], [], []
        for chunk in chunks:
            if not chunk:
                chunk = ["[PAD]"]
            encoded = self.tokenizer.encode_plus(
                chunk,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'].squeeze(0))
            attention_masks.append(encoded['attention_mask'].squeeze(0))
            token_type_ids.append(encoded['token_type_ids'].squeeze(0))
        return torch.stack(input_ids), torch.stack(attention_masks), torch.stack(token_type_ids)
     
    def __getitem__(self, index):
        text = self.text[index]
        prompt = self.prompt[index].replace('\n', ' ')
        combined_text = f"[prompt] {prompt} [prompt] {prompt} [ESSAY] {text}"
        chunks = self.segment_text(combined_text)
        input_ids_doc, attention_mask_doc, token_type_ids_doc = self.process_chunks(chunks)
        return {
            'input_ids': input_ids_doc,
            'attention_mask': attention_mask_doc,
            'token_type_ids': token_type_ids_doc,
            'labels': torch.FloatTensor(self.labels[index])
        }           
