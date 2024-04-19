import torch
import torch.nn.functional as F

class CombinedLoss(torch.nn.Module):
    def __init__(self, weights=[0.33, 0.33, 0.33], b=0.0):
        super(CombinedLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.weights = weights
        self.b = b

    def forward(self, y, y_hat):
        # Compute MSE
        mse_loss_val = self.mse_loss(y, y_hat)

        # Compute similarity loss; the result is a single-element tensor
        sim_loss_val = (1 - F.cosine_similarity(y.unsqueeze(0), y_hat.unsqueeze(0)))

        # Compute margin ranking loss
        y_greater = (y > y_hat).float()
        y_less = (y < y_hat).float()
        y_equal = (y == y_hat).float()
        r_ij = y_greater - y_less - y_equal * torch.sign(y - y_hat)
        mr_loss_val = (r_ij * (y - y_hat) + self.b).clamp(min=0).mean()

        # Combine the losses by averaging with weights
        # Make sure to extract the scalar value from sim_loss_val tensor
        combined_loss_val = mse_loss_val * self.weights[0] + sim_loss_val.item() * self.weights[1] + mr_loss_val * self.weights[2]     
        return combined_loss_val

from torch import nn
import torch

class multi_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = [40, 100, 1]
        self.device = 'cuda'
        self.MSE = nn.MSELoss().to(device=self.device)
        self.CosineEmbeddingLoss = nn.CosineEmbeddingLoss().to(device=self.device)
        self.MarginRankingLoss = nn.MarginRankingLoss().to(device=self.device)

    def forward(self, y_trues, y_preds):
        """
        input must be [batchsize, 1]
        """
        m, n = y_trues.size()
        batchsize = y_preds.shape[0]
        mseloss = self.MSE(y_trues, y_preds)
        simloss = torch.max(torch.tensor(0., device=self.device), self.CosineEmbeddingLoss(y_trues.resize(n, m), y_preds.resize(n, m), torch.ones(batchsize, dtype=torch.int, device=self.device)))

        # count rankloss
        rankloss = torch.tensor(0., device=self.device)
        for i in range(batchsize):
            for j in range(i + 1, batchsize):
                input1_pred = y_preds[i]
                input2_pred = y_preds[j]
                input1_true = y_trues[i]
                input2_true = y_trues[j]
                target = 0
                if input1_true > input2_true:
                    target = 1
                elif input1_true < input2_true:
                    target = -1
                else:
                    if input1_pred > input2_pred:
                        target = -1
                    elif input1_pred < input2_pred:
                        target = 1
                target = torch.tensor([target], device=self.device)
                rankloss += self.MarginRankingLoss(input1_pred, input2_pred, target)

        print(f'mseloss{self.weight[0] * mseloss}\tsimloss{self.weight[1] * simloss}\trankloss{self.weight[2] * rankloss}')

        return self.weight[0] * mseloss + self.weight[1] * simloss + self.weight[2] * rankloss
