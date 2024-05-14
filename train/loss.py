import torch
import torch.nn.functional as F

class CombinedLoss(torch.nn.Module):
    def __init__(self, weights=[0.6, 0.4], b=0.0):
        super(CombinedLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
        # self.smoothloss = torch.nn.SmoothL1Loss()
        self.weights = weights
        self.b = b

    def forward(self, y, y_hat):
        mse_loss = self.mse_loss(y, y_hat)
        sim_loss_val = (1 - F.cosine_similarity(y.unsqueeze(0), y_hat.unsqueeze(0)))
        combined_loss_val = mse_loss * self.weights[0] + sim_loss_val.item() * self.weights[1]    
        return combined_loss_val