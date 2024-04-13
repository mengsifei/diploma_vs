import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

def evaluate_model(model, loader, criteria, past_losses, is_dual_version, device):
    model.eval()
    running_losses = [0.0] * 4
    total_weights = [0.0] * 4
    all_preds = []
    all_targets = []

    average_past_losses = [np.mean(losses) if losses else 1.0 for losses in past_losses]
    task_weights = [1.0 / (loss + 1e-6) for loss in average_past_losses]

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k.endswith('_ids') or k.endswith('_mask')}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            label_weights = batch['label_weights'].to(device).squeeze(1)
            preds = outputs.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels_np)

            batch_size = labels.size(0)
            for i in range(4):
                weighted_loss = criteria[i](outputs[:, i], labels[:, i])
                weighted_loss *= label_weights[:, i]
                final_loss = weighted_loss.mean() #* task_weights[i]
                running_losses[i] += final_loss.item() * batch_size
                total_weights[i] += label_weights[:, i].sum().item()

        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        kappas = [cohen_kappa_score(np.round(all_targets[:, i]).astype(int), np.round(all_preds[:, i]).astype(int), weights='quadratic') for i in range(4)]
        maes = [mean_absolute_error(all_targets[:, i], all_preds[:, i]) for i in range(4)]
        avg_mse_losses = [running_loss / total_weights[i] if total_weights[i] != 0 else 0 for i, running_loss in enumerate(running_losses)]

        print("============Average MSE Losses on Validation=============")
        for i, mse_loss in enumerate(avg_mse_losses):
            print(f" Subtask {i+1}: {mse_loss:.4f}")
        print("============MAEs per Criterion=============\n", maes)
        print("============Quadratic Weighted Cohen Kappa Scores per Criterion=============\n", kappas)
        
        return maes, kappas, avg_mse_losses
