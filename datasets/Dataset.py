import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512, model_type='model1'):
        self.tokenizer = tokenizer
        self.df = df
        self.text = df['essay']
        self.prompt = df['prompt']
        self.max_len = max_len
        self.model_type = model_type
        
        # Split the labels based on the model type
        if model_type == 'model1':
            # For the first model, which predicts Task Response and Coherence and Cohesion
            self.labels = df[['Task Response']].values #, 'Coherence and Cohesion'
        elif model_type == 'model2':
            # For the second model, which predicts Lexical Resource and Grammatical Range and Accuracy
            self.labels = df[['Lexical Resource', 'Grammatical Range and Accuracy']].values
        else:
            raise ValueError("Invalid model_type specified. Use 'model1' or 'model2'.")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        prompt = self.prompt[index]
        combined_text = f"[prompt] {prompt} [ESSAY] {text}"
        # Process text to handle special tokens, padding, and return a tensor
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
    
