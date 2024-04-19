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
    def __getitem__(self, index):
        text = self.text[index]
        prompt = self.prompt[index]
        combined_text = f"[prompt] {prompt} [ESSAY] {text}"
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

import torch
from torch.utils.data import Dataset

class CustomDatasetChunk(Dataset):
    def __init__(self, df, tokenizer, max_len=512, segments=[80, 150, 150, 80], max_chunks=None):
        self.tokenizer = tokenizer
        self.df = df
        self.text = df['essay']
        self.topic = df['prompt']
        self.labels = self.df[['Task Response', 'Coherence and Cohesion', 
                               'Lexical Resource', 'Grammatical Range and Accuracy']].values
        self.max_len = max_len
        self.segment_lengths = segments
        self.max_chunks = max_chunks if max_chunks is not None else len(segments)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        topic = self.topic[index]
        combined_text = f"[TOPIC] {topic} [ESSAY] {text}"
        tokenized_text = self.tokenizer.tokenize(combined_text)
        
        # Process document-level inputs uniformly
        input_ids_doc, attention_mask_doc, token_type_ids_doc = self.process_chunks(tokenized_text, self.max_len, self.max_chunks)

        # Process segments based on predefined lengths
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
        chunk_size = max_len - 2  # account for [CLS] and [SEP]
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
                chunk = ["[UNK]"]  # Fallback for empty chunks
            encoded = self.tokenizer.encode_plus(
                chunk, add_special_tokens=True, max_length=max_len,
                padding='max_length', truncation=True, return_tensors='pt')
            input_ids.append(encoded['input_ids'].squeeze(0))
            attention_masks.append(encoded['attention_mask'].squeeze(0))
            token_type_ids.append(encoded['token_type_ids'].squeeze(0))

        # Pad remaining chunks if necessary
        while len(input_ids) < max_chunks:
            input_ids.append(torch.zeros(max_len, dtype=torch.long))
            attention_masks.append(torch.zeros(max_len, dtype=torch.long))
            token_type_ids.append(torch.zeros(max_len, dtype=torch.long))

        return torch.stack(input_ids), torch.stack(attention_masks), torch.stack(token_type_ids)


        
           

# class CustomDatasetSegment(torch.utils.data.Dataset):
#     def __init__(self, df, tokenizer, max_len=512):
#         self.tokenizer = tokenizer
#         self.df = df
#         self.text = df['essay']
#         self.prompt = df['prompt']
#         self.labels = self.df[['Task Response', 'Coherence and Cohesion',
#        'Lexical Resource', 'Grammatical Range and Accuracy']].values
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.text)
#     def calculate_features(self, text):
#         # word_pattern = re.compile(r'\w+')
#         paragraph_pattern = re.compile(r'\n')
#         # sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
#         # words = word_pattern.findall(text)
#         # num_words = len(words)
#         num_paragraphs = len(paragraph_pattern.findall(text)) + 1
#         # num_sentences = len(sentence_pattern.findall(text)) + 1
#         # stop_words = set(stopwords.words('english'))
#         # word_counts = Counter(word.lower() for word in words if word.lower() not in stop_words)
#         # frequent_words = sum(1 for _, count in word_counts.items() if count > 3)
#         features = np.array([num_paragraphs], dtype=np.float32)
#         # mean_features = np.mean(features)
#         # std_features = np.std(features)
#         # normalized_features = self.z_score_normalize(features, mean_features, std_features)
#         return 1 / features
#     def __getitem__(self, index):
#         text = self.text[index]
#         prompt = self.prompt[index]
#         inputs = self.tokenizer.encode_plus(
#             [prompt.replace('\n', ''), text.replace("\n", f" [SEP][SEP] ")],
#             None,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding='max_length',
#             return_token_type_ids=True,
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': inputs['input_ids'].flatten(),
#             'attention_mask': inputs['attention_mask'].flatten(),
#             'token_type_ids': inputs['token_type_ids'].flatten(),
#             'labels': torch.FloatTensor(list(self.labels[index]))
#         }


# class CustomDatasetDual(torch.utils.data.Dataset):
#     def __init__(self, df, tokenizer, max_len=512):
#         self.tokenizer = tokenizer
#         self.df = df
#         self.essays = df['essay']
#         self.prompts = df['prompt']
#         self.labels = df[['Task Response', 'Coherence and Cohesion', 'Lexical Resource', 'Grammatical Range and Accuracy']].values
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.essays)

#     def __getitem__(self, index):
#         essay_text = self.essays[index]
#         prompt_text = self.prompts[index]
#         essay_text = essay_text.replace("\n", f" [SEP] ")
#         essay_inputs = self.tokenizer(
#             essay_text,
#             max_length=self.max_len - 60, 
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
        
#         prompt_inputs = self.tokenizer(
#             prompt_text,
#             max_length=60,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
        
#         return {
#             'essay_input_ids': essay_inputs['input_ids'].flatten(),
#             'essay_attention_mask': essay_inputs['attention_mask'].flatten(),
#             'prompt_input_ids': prompt_inputs['input_ids'].flatten(),
#             'prompt_attention_mask': prompt_inputs['attention_mask'].flatten(),
#             'labels': torch.FloatTensor(self.labels[index])
#         }
    
