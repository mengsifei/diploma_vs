import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

def evaluate_model(model, loader, criteria, device):
    model.eval()  # Set the model to evaluation mode
    total_mse_losses = [0, 0, 0, 0]
    all_preds = []
    all_targets = []
    
    with torch.no_grad():  # No gradients needed during evaluation
        for batch in loader:
            inputs = {k: batch[k].to(device) for k in ['input_ids', 'attention_mask']}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            preds = outputs.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels_np)
            
            # Calculate the loss for each subtask
            for i in range(4):  # Assuming 4 subtasks
                loss = criteria[i](outputs[:, i], labels[:, i])
                total_mse_losses[i] += loss.item()

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics for each subtask
    kappas = []
    maes = []
    for i in range(4):  # Assuming 4 subtasks
        # Compute Cohen's Kappa and MAE for each subtask
        kappa = cohen_kappa_score(np.round(all_targets[:, i] * 2).astype(int), 
                                  np.round(all_preds[:, i] * 2).astype(int), 
                                  weights='quadratic')
        mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        kappas.append(kappa)
        maes.append(mae)
    
    avg_mse_losses = [total / len(loader) for total in total_mse_losses]
    print("============Average MSE Losses on Validation=============")
    for i, mse_loss in enumerate(avg_mse_losses):
        print(f" Subtask {i+1}: {mse_loss:.4f}")
    print("============MAEs per Criterion=============\n", maes)
    print("============Quadratic Weighted Cohen Kappa Scores per Criterion=============\n", kappas)
    
    return maes, kappas, avg_mse_losses