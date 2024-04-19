import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class mainplm(nn.Module):
    def __init__(self):
        super(mainplm, self).__init__()
        self.plm_batch_size = 1
        self.plm = AutoModel.from_pretrained('google/electra-small-discriminator')

        for param in self.plm.embeddings.parameters():
            param.requires_grad = False
        for i in range(11):
            for param in self.plm.encoder.layer[i].parameters():
                param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.plm.config.hidden_size, 4)
        )
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor):
        all_plm_output = self.plm(
            document_batch[:, 0:self.plm_batch_size, 0].view(-1, document_batch.size(-1)),  # Flatten the input
            token_type_ids=document_batch[:, 0:self.plm_batch_size, 1].view(-1, document_batch.size(-1)),
            attention_mask=document_batch[:, 0:self.plm_batch_size, 2].view(-1, document_batch.size(-1))
        )
        
        # Directly extract the relevant hidden state instead of using a loop
        plm_output = all_plm_output.last_hidden_state[:, 0, :].view(document_batch.shape[0], -1)
        
        prediction = self.mlp(plm_output)
        return prediction

class chunkplm(nn.Module):
    def __init__(self):
        super(chunkplm, self).__init__()
        self.plm = AutoModel.from_pretrained('google/electra-small-discriminator')

        for param in self.plm.embeddings.parameters():
            param.requires_grad = False
        for i in range(12):
            for param in self.plm.encoder.layer[i].parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(self.plm.config.hidden_size, self.plm.config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.plm.config.hidden_size, 1)
        )
        self.w_omega = nn.Parameter(torch.Tensor(self.plm.config.hidden_size, self.plm.config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, self.plm.config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(self.plm.config.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, plm_batch_size=0):
        # Perform PLM operations on all chunks at once
        plm_output = self.plm(
            document_batch.view(-1, document_batch.size(-1)),  # Flatten the input
            attention_mask=(document_batch != self.tokenizer.pad_token_id).view(-1, document_batch.size(-1))
        )[1].view(document_batch.shape[0], plm_batch_size, -1)
        
        # Use built-in LSTM, no need for manual looping
        output, _ = self.lstm(plm_output)
        
        # Attention calculations can be done on batch tensors directly
        attention_w = torch.tanh(torch.matmul(output, self.w_omega) + self.b_omega)
        attention_score = F.softmax(attention_w.matmul(self.u_omega), dim=1)
        attention_hidden = (output * attention_score).sum(dim=1)
        
        prediction = self.mlp(attention_hidden)
        return prediction

import torch
from torch import nn, optim
from transformers import AutoModel, AutoTokenizer
from train.loss import * 
import numpy as np
from torch.utils.data import DataLoader

class AESmodel(nn.Module):
    def __init__(self, train_data, val_data, device):
        super(AESmodel, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')
        self.chunk_sizes = [int(size) for size in '90_30_130_10'.split('_') if size != "0"]
        self.bert_batch_sizes = [int(649 / chunk_size) + 1 for chunk_size in self.chunk_sizes]
        
        self.bert_regression_by_word_document = mainplm()
        self.bert_regression_by_chunk = chunkplm()

        self.optim = optim.Adam([
            {'params': self.mainplm.parameters(), 'lr': 1e-4},
            {'params': self.chunkplm.parameters(), 'lr': 1e-4}
        ])
       
        self.train_data = DataLoader(train_data, batch_size=32, shuffle=True)
        self.val_data = DataLoader(val_data, batch_size=32, shuffle=False)
        
        self.multi_loss = multi_loss()
        
        self.best_val_qwk = [-np.inf] * 4
        self.best_val_mae = [np.inf] * 4


    def forward(self, inputs):
        # Assuming inputs are preprocessed properly
        mainplm_output = self.mainplm(inputs)
        chunkplm_output = self.chunkplm(inputs)
        # Combine outputs from both models
        output = mainplm_output + chunkplm_output
        return output

    def train(self, epoch):
        traindata = self.train_data
        self.bert_regression_by_word_document.to(device=self.device)
        self.bert_regression_by_chunk.to(device=self.device)
        self.multi_loss.to(device=self.device)
        for e in range(epoch):
            print('*' * 20 + f'epoch: {e + 1}' + '*' * 20)
            self.bert_regression_by_word_document.train()
            self.bert_regression_by_chunk.train()
            target_scores = None
            if isinstance(traindata, tuple) and len(traindata) == 2:
                doctok_token_indexes, doctok_token_indexes_slicenum = encode_documents(
                    traindata[0], self.tokenizer, max_input_length=512)
                # [document_number:144, 510times:3, 3, bert_len:512] [每document有多少510:144]
                # traindata[0] is the essays
                chunk_token_indexes_list, chunk_token_indexes_length_list = [], []
                for i in range(len(self.chunk_sizes)): # 以固定的chunk划分
                    document_representations_chunk, document_sequence_lengths_chunk = encode_documents(
                        traindata[0],
                        self.tokenizer,
                        max_input_length=self.chunk_sizes[i])
                    chunk_token_indexes_list.append(document_representations_chunk)
                    chunk_token_indexes_length_list.append(document_sequence_lengths_chunk)
                target_scores = torch.FloatTensor(traindata[1])


            predictions = torch.empty((doctok_token_indexes.shape[0]))
            acculation_loss = 0.
            for i in range(0, doctok_token_indexes.shape[0], self.args['batch_size']): # range(0, 144, 32)
                self.optim.zero_grad()
                batch_doctok_token_indexes = doctok_token_indexes[i:i + self.args['batch_size']].to(device=self.device)
                batch_target_scores = target_scores[i:i + self.args['batch_size']].to(device=self.device)
                with autocast():
                    batch_doctok_predictions = self.bert_regression_by_word_document(batch_doctok_token_indexes, device=self.device)
                batch_doctok_predictions = torch.squeeze(batch_doctok_predictions)


                batch_predictions = batch_doctok_predictions
                # for chunk_index in range(len(self.chunk_sizes)):
                #     batch_document_tensors_chunk = chunk_token_indexes_list[chunk_index][i:i + self.args['batch_size']].to(
                #         device=self.device)
                #     batch_predictions_chunk = self.bert_regression_by_chunk(
                #         batch_document_tensors_chunk,
                #         device=self.device,
                #         plm_batch_size=self.bert_batch_sizes[chunk_index]
                #     )
                #     batch_predictions_chunk = torch.squeeze(batch_predictions_chunk)
                #     batch_predictions = torch.add(batch_predictions, batch_predictions_chunk) # 多个chunk的分加起来

                if len(batch_predictions.shape) == 0: # 证明只有一个tensor，不构成list
                    batch_predictions = torch.tensor([batch_predictions], device=self.device)
                with autocast():
                    loss = self.multi_loss(batch_target_scores.unsqueeze(1), batch_predictions.unsqueeze(1))
                
                
                
                
                loss.requires_grad_(True)
                # loss.backward()
                scaler.scale(loss).backward()
                
                # self.optim.step()
                scaler.step(self.optim)
                scaler.update()
                acculation_loss += loss.item()

                predictions[i:i + self.args['batch_size']] = batch_predictions
            assert target_scores.shape == predictions.shape

            print(f'epoch{e + 1} avg loss is {acculation_loss / doctok_token_indexes.shape[0]}')
            # 到此已获得predictions
            prediction_scores = []
            label_scores = []
            predictions = predictions.detach().numpy()
            target_scores = target_scores.detach().numpy()

            for index, item in enumerate(predictions):
                prediction_scores.append(fix_score(item, self.prompt))
                label_scores.append(target_scores[index])

            train_eva_res = evaluation(label_scores, prediction_scores)
            df = pd.DataFrame(dict(zip(['prediction', 'prediction_fix', 'target'], [predictions.tolist(), prediction_scores, label_scores])))
            df.to_csv(f'./prediction/p{self.prompt}/{self.foldname}/train/{e + 1}_pred.csv', index=False)
            print('-' * 10 + 'trainset' + '-' * 10)
            print("pearson:", float(train_eva_res[7]))
            print("qwk:", float(train_eva_res[8]))
            self.plt_x.append(e + 1)
            self.plt_train_qwk.append(float(train_eva_res[8]))
            self.validate(self.valdata, e, mode='val')
            self.validate(self.testdata, e, mode='test')
            plt.plot(self.plt_x, self.plt_train_qwk, 'ro-', color='blue', alpha=0.8, linewidth=1, label='train')
            plt.plot(self.plt_x, self.plt_val_qwk, 'ro-', color='yellow', alpha=0.8, linewidth=1, label='val')
            plt.plot(self.plt_x, self.plt_test_qwk, 'ro-', color='red', alpha=0.8, linewidth=1, label='test')
            plt.title(self.foldname)
            plt.xlabel('epoch')
            plt.ylabel('qwk')
            plt.legend(loc='lower right')
            plt.savefig(f'./prediction/p{self.prompt}/{self.foldname}/qwk.jpg')
            plt.close()