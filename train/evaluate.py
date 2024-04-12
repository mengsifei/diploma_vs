import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

def evaluate_model(model, loader, criteria, is_dual_version, device):
    model.eval()
    running_losses = [0.0] * 4
    total_weights = [0.0] * 4
    loss_weights = [0.25] * 4
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            if is_dual_version:
                inputs = {
                    'essay_input_ids': batch['essay_input_ids'].to(device),
                    'essay_attention_mask': batch['essay_attention_mask'].to(device),
                    'topic_input_ids': batch['topic_input_ids'].to(device),
                    'topic_attention_mask': batch['topic_attention_mask'].to(device)
                }
            else:
                inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            label_weights = batch['label_weights'].to(device).squeeze(1)
            preds = outputs.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels_np)

            batch_size = labels.size(0)
            losses = []
            for i in range(4):  # Assuming 4 subtasks
                weighted_loss = criteria[i](outputs[:, i], labels[:, i])
                weighted_loss *= label_weights[:, i]  # Apply weights element-wise
                final_loss = weighted_loss.mean() * loss_weights[i]
                losses.append(final_loss)
                running_losses[i] += final_loss.item() * batch_size
                total_weights[i] += label_weights[:, i].sum().item()

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    kappas = [cohen_kappa_score(np.round(all_targets[:, i]).astype(int), np.round(all_preds[:, i]).astype(int), weights='quadratic') for i in range(4)]
    maes = [mean_absolute_error(all_targets[:, i], all_preds[:, i]) for i in range(4)]
    avg_mse_losses = [total / total_weights[i] for i, total in enumerate(running_losses)]

    print("============Average MSE Losses on Validation=============")
    for i, mse_loss in enumerate(avg_mse_losses):
        print(f" Subtask {i+1}: {mse_loss:.4f}")
    print("============MAEs per Criterion=============\n", maes)
    print("============Quadratic Weighted Cohen Kappa Scores per Criterion=============\n", kappas)
    
    return maes, kappas, avg_mse_losses
