
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

def evaluate_model(model, loader, criteria, device, rubrics):
    model.eval()  # Set the document model to evaluation mode
    running_losses = {rubric: 0.0 for rubric in rubrics}
    task_weights = [0.25, 0.25, 0.2, 0.3]
    all_preds = []
    all_targets = []
    total_samples = 0  # This will store the total samples processed per task

    with torch.no_grad():
        for batch in loader:
            labels = batch['labels'].to(device)  # Assuming labels are the same for both batch
            outputs = model(batch['input_ids'].to(device),
                                    batch['attention_mask'].to(device),
                                    batch['token_type_ids'].to(device))
            preds = outputs.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels_np)
            batch_size = labels.size(0)
            losses = {rubrics[i]: criteria[i](outputs[:, i], labels[:, i]) * task_weights[i] for i in range(len(rubrics))}
            for rubric in rubrics:
                running_losses[rubric] += losses[rubric].item() * labels.size(0)
            total_samples += labels.size(0)

        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        kappas = [cohen_kappa_score(np.round(all_targets[:, i]).astype(int), np.round(all_preds[:, i]).astype(int), weights='quadratic') for i in range(len(rubrics))]
        maes = [mean_absolute_error(all_targets[:, i], all_preds[:, i]) for i in range(len(rubrics))]
        
        avg_mse_losses = [running_losses[rubric] / total_samples for rubric in running_losses]

        print("============Average MSE Losses on Validation=============")
        for i, mse_loss in enumerate(avg_mse_losses):
            print(f" Subtask {i+1}: {mse_loss:.4f}")
        print("============MAEs per Criterion=============\n", maes)
        print("============Quadratic Weighted Cohen Kappa Scores per Criterion=============\n", kappas)
        
        return maes, kappas, avg_mse_losses
