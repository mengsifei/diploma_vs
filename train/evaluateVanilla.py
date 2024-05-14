import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

def evaluate_model(model, loader, criteria, device, rubrics, task_weights = [0.25, 0.25, 0.25, 0.25]):
    model.eval()
    running_losses = [0.0] * len(rubrics)
    all_preds = []
    all_targets = []
    total_samples = [0] * len(rubrics)

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k.endswith('_ids') or k.endswith('_mask')}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            preds = outputs.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels_np)

            batch_size = labels.size(0)
            for i in range(len(rubrics)):
                weighted_loss = criteria[i](outputs[:, i], labels[:, i])
                final_loss = weighted_loss * task_weights[i]
                running_losses[i] += final_loss.item() * batch_size
                total_samples[i] += batch_size

        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        kappas = [cohen_kappa_score(np.round(all_targets[:, i]).astype(int), np.round(all_preds[:, i]).astype(int), weights='quadratic') for i in range(len(rubrics))]
        maes = [mean_absolute_error(all_targets[:, i], all_preds[:, i]) for i in range(len(rubrics))]
        
        avg_mse_losses = [running_loss / total_sample for running_loss, total_sample in zip(running_losses, total_samples)]

        print("============Average MSE Losses on Validation=============")
        for i, mse_loss in enumerate(avg_mse_losses):
            print(f" Subtask {i+1}: {mse_loss:.4f}")
        print("============MAEs per Criterion=============\n", maes)
        print("============Quadratic Weighted Cohen Kappa Scores per Criterion=============\n", kappas)
        
        return maes, kappas, avg_mse_losses

