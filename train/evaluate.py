import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

def evaluate_model(model, loader, criteria, device, rubrics):
    model.eval()
    running_losses = [0.0] * len(rubrics)
    task_weights = [1 / len(rubrics)] * len(rubrics)
    all_preds = []
    all_targets = []
    total_samples = [0] * len(rubrics)  # This will store the total samples processed per task

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
                total_samples[i] += batch_size  # Update total samples for each task

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


def evaluate_model_chunk(models, loader, criteria, device, rubrics):
    model_doc, model_chunk = models  # Unpack the tuple into separate models
    model_doc.eval()  # Set the document model to evaluation mode
    model_chunk.eval()  # Set the chunk model to evaluation mode

    running_losses = [0.0] * len(rubrics)
    task_weights = [1 / len(rubrics)] * len(rubrics)
    all_preds = []
    all_targets = []
    total_samples = [0] * len(rubrics)  # This will store the total samples processed per task

    with torch.no_grad():
        for batch in loader:
            doc_inputs, seg_inputs = batch  # Assume batch unpacks into document and segment inputs
            labels = doc_inputs['labels'].to(device)  # Assuming labels are the same for both inputs

            # Process document-level inputs
            doc_outputs = model_doc(doc_inputs['input_ids'].to(device),
                                    doc_inputs['attention_mask'].to(device),
                                    doc_inputs['token_type_ids'].to(device))

            # Process segment-level inputs
            chunk_outputs = []
            for seg_idx in range(len(seg_inputs['input_ids'])):
                seg_output = model_chunk(seg_inputs['input_ids'][seg_idx].to(device),
                                         seg_inputs['attention_mask'][seg_idx].to(device),
                                         seg_inputs['token_type_ids'][seg_idx].to(device))
                chunk_outputs.append(seg_output)

            # Combine outputs from all segments
            combined_chunk_output = torch.mean(torch.stack(chunk_outputs), dim=0)

            # Combine document and chunk outputs
            final_outputs = (doc_outputs + combined_chunk_output) / 2
            preds = final_outputs.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels_np)

            batch_size = labels.size(0)
            for i in range(len(rubrics)):
                weighted_loss = criteria[i](final_outputs[:, i], labels[:, i])
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
