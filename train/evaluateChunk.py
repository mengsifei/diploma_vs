
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

def evaluate_model_chunk(models, loader, criteria, device, rubrics):
    model_doc, model_chunk = models 
    model_doc.eval()
    model_chunk.eval()

    running_losses = {rubric: 0.0 for rubric in rubrics}
    task_weights = [1 / len(rubrics)] * len(rubrics)
    all_preds = []
    all_targets = []
    total_samples = 0 
    with torch.no_grad():
        for batch in loader:
            doc_inputs, seg_inputs = batch
            labels = doc_inputs['labels'].to(device)  
            doc_outputs = model_doc(doc_inputs['input_ids'].to(device),
                                    doc_inputs['attention_mask'].to(device),
                                    doc_inputs['token_type_ids'].to(device))
            chunk_outputs = model_chunk(seg_inputs['input_ids'].to(device),
                                    seg_inputs['attention_mask'].to(device),
                                    seg_inputs['token_type_ids'].to(device), device)
           
            final_outputs = (doc_outputs + chunk_outputs) / 2
            preds = final_outputs.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels_np)

            batch_size = labels.size(0)
            losses_doc = {rubrics[i]: criteria[0][i](doc_outputs[:, i], labels[:, i]) * task_weights[i] for i in range(len(rubrics))}
            losses_chunk = {rubrics[i]: criteria[1][i](chunk_outputs[:, i], labels[:, i]) * task_weights[i] for i in range(len(rubrics))}

            for rubric in rubrics:
                running_losses[rubric] += (losses_doc[rubric].item() + losses_chunk[rubric].item()) / 2 * labels.size(0)
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
