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


# def forward(self, input_ids, attention_mask, token_type_ids=None):
#     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#     last_hidden_state = outputs.last_hidden_state
#     pooled_output = self.pooler(last_hidden_state)
#     dropout_output = self.dropout(pooled_output)
#     final_outputs = self.out(dropout_output)  # This is the logits output for each class
#     return final_outputs

# def __getitem__(self, index):
#         text = self.text[index].replace('\n', '[SEP]')
#         prompt = self.prompt[index]
#         combined_text = f"[prompt] {prompt} [prompt] {prompt} [ESSAY] {text}"
#         tokenized_text = self.tokenizer.tokenize(combined_text)
#         input_ids_doc, attention_mask_doc, token_type_ids_doc = self.process_chunks(tokenized_text, self.max_len, self.max_chunks)
#         inputs = self.tokenizer.encode_plus(
#             combined_text,
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
#             'labels': torch.FloatTensor(self.labels[index])
#         }  