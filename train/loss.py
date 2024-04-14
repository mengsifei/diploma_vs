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

