import torch
from torch.utils.data import Dataset

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
    def process_chunks(self, tokens, max_len=512, max_chunks=3):
        chunk_size = max_len - 2
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        return self.encode_chunks(chunks, max_len, max_chunks)
    def encode_chunks(self, chunks, max_len, max_chunks):
        input_ids, attention_masks, token_type_ids = [], [], []
        for chunk in chunks:
            if not chunk:
                chunk = ["[PAD]"]
            encoded = self.tokenizer.encode_plus(
                chunk,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                truncation=False,
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
    
    def __getitem__(self, index):
        text = self.text[index].replace('\n', '[SEP]')
        prompt = self.prompt[index]
        combined_text = f"[prompt] {prompt} [prompt] {prompt} [ESSAY] {text}"
        tokenized_text = self.tokenizer.tokenize(combined_text)
        input_ids_doc, attention_mask_doc, token_type_ids_doc = self.process_chunks(tokenized_text, self.max_len, self.max_chunks)
        return {
            'input_ids': input_ids_doc,
            'attention_mask': attention_mask_doc,
            'token_type_ids': token_type_ids_doc,
            'labels': torch.FloatTensor(self.labels[index])
        }           
