import torch
from torch.utils.data import Dataset

class CustomDatasetChunk(Dataset):
    def __init__(self, df, tokenizer, max_len=512, segments=[80, 150, 150, 80], max_chunks=None):
        self.tokenizer = tokenizer
        self.df = df
        self.text = df['essay'].tolist()
        self.topic = df['prompt'].tolist()
        self.labels = self.df[['Task Response', 'Coherence and Cohesion', 
                               'Lexical Resource', 'Grammatical Range and Accuracy']].values.tolist()
        self.max_len = max_len
        self.segment_lengths = segments
        self.max_chunks = max_chunks if max_chunks is not None else len(segments)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        topic = self.topic[index]
        combined_text = f"[prompt] {topic} [prompt] {topic} [ESSAY] {text}"
        tokenized_text = self.tokenizer.tokenize(combined_text)
        
        input_ids_doc, attention_mask_doc, token_type_ids_doc = self.process_chunks(tokenized_text, self.max_len, self.max_chunks)
        input_ids_seg, attention_mask_seg, token_type_ids_seg = self.process_fixed_segments(tokenized_text, self.segment_lengths)

        return [{
                'input_ids': input_ids_doc,
                'attention_mask': attention_mask_doc,
                'token_type_ids': token_type_ids_doc,
                'labels': torch.FloatTensor(self.labels[index])
            }, {
                'input_ids': input_ids_doc,
                'attention_mask': attention_mask_doc,
                'token_type_ids': token_type_ids_doc,
                'labels': torch.FloatTensor(self.labels[index])
            }]

    def process_chunks(self, tokens, max_len, max_chunks):
        chunk_size = max_len - 2
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        return self.encode_chunks(chunks, max_len, max_chunks)

    def process_fixed_segments(self, tokens, segment_lengths):
        pointer = 0
        segments = []
        for length in segment_lengths:
            if pointer + length > len(tokens):
                break
            segments.append(tokens[pointer:pointer + length])
            pointer += length
        return self.encode_chunks(segments, max(self.segment_lengths), len(segment_lengths))

    def encode_chunks(self, chunks, max_len, max_chunks):
        input_ids, attention_masks, token_type_ids = [], [], []
        for chunk in chunks:
            if not chunk:
                chunk = ["[UNK]"]
            encoded = self.tokenizer.encode_plus(
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


 