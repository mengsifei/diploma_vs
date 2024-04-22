import torch
class CustomDatasetDual(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.df = df
        self.essays = df['essay']
        self.prompts = df['prompt']
        self.labels = df[['Task Response', 'Coherence and Cohesion', 'Lexical Resource', 'Grammatical Range and Accuracy']].values
        self.max_len = max_len

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, index):
        essay_text = self.essays[index]
        prompt_text = self.prompts[index]
        essay_text = essay_text.replace("\n", f" [SEP] ")
        essay_inputs = self.tokenizer.encode_plus(
            essay_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len - 40,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        prompt_inputs = self.tokenizer(
            essay_text,
            prompt_text,
            add_special_tokens=True,
            max_length= 40,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'essay_input_ids': essay_inputs['input_ids'].flatten(),
            'essay_attention_mask': essay_inputs['attention_mask'].flatten(),
            'essay_token_type_ids': essay_inputs['token_type_ids'].flatten(),
            'prompt_input_ids': prompt_inputs['input_ids'].flatten(),
            'prompt_attention_mask': prompt_inputs['attention_mask'].flatten(),
            'prompt_token_type_ids': prompt_inputs['token_type_ids'].flatten(),
            'labels': torch.FloatTensor(self.labels[index])
        }
    
       
