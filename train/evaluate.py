import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

def evaluate_model(model, loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_mse_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():  # No gradients needed
        for batch in loader:
            inputs = {k: batch[k].to(device) for k in ['input_ids', 'attention_mask']}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            loss = criterion(outputs, labels.float())
            total_mse_loss += loss.item()
            
            preds = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(labels_np)
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics for each criterion
    kappas = []
    maes = []
    for i in range(4):  # Assuming 4 criteria
        kappa = cohen_kappa_score((all_targets[:, i] * 10).astype(int), (np.round(all_preds[:, i] * 2) * 5).astype(int), weights='quadratic')
        mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        kappas.append(kappa)
        maes.append(mae)
    avg_mse_loss = total_mse_loss / len(loader.dataset)
    print(f"Average MSE Loss: {avg_mse_loss}")
    print("MAEs per Criterion:")
    print(maes)
    print("Quadratic Weighted Cohen Kappa Scores per Criterion")
    print(kappas)
    return maes, kappas, avg_mse_loss